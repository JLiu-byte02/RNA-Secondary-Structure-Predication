"""src/data.py -- canonical RNA secondary-structure dataset, tokenization, batching.

Canonical record format (produced by the DATA task, one gzipped JSONL per split)::

    {"id": str, "sequence": str, "pairs": [[i0, j0], ...],   # 0-based, i<j
     "length": int, "split": str, "family": str, "source_id": str}

This module is the single place where *file* formats become *tensors*:

* sequence normalisation (``T -> U``) and sha256 keying (thermo caches),
* RiNALMo ``Alphabet`` tokenization (lazy import; simple ACGU fallback),
* padded token batch + ``L x L`` pair target + candidate mask + valid pair mask,
* optional thermodynamic channels (BPP / stack / distance / mask) from per-sample
  ``.npz`` files keyed by ``sha256(normalized_sequence)``,
* length-bucketed batching so batch padding stays small on 3090-class GPUs.

Index convention: **files are 0-based already** (unlike the prior project, which
stored 1-based pair strings).  Nothing in this module converts back to 1-based;
export is handled by ``scripts/evaluate.py`` at write time if ever needed.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# ---------------------------------------------------------------------------
# Pair rules -- mirrored from H:\RNA-Secondary-Structure-Predication\src\utils.py
# and from RiNALMo's rinalmo/utils/sec_struct.py (canonical pairs, no sharp loops)
# ---------------------------------------------------------------------------
CANONICAL_PAIRS = frozenset(
    {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
)
MIN_SEP = 4  # |i - j| >= 4  <=>  no pairs within a 3-nt loop ("sharp loop" mask)
RNA_ALPHABET = "ACGU"

# RiNALMo token table (rinalmo/data/constants.py), kept local so that the
# fallback alphabet is byte-identical to the real one when rinalmo is absent.
_SPECIAL_TOKENS = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
_RNA_TOKENS = ["A", "C", "G", "T", "I", "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", "N", "-"]

_CHANNEL_ALIASES = {
    "bpp": ("bpp", "BPP", "P", "P_bpp", "bpp_matrix"),
    "stack": ("stack", "Stack", "Q", "Q_stack", "stack_matrix"),
    "mfe": ("mfe_dbn", "mfe", "MFE"),
    "mea": ("mea_dbn", "mea", "MEA"),
    "centroid": ("centroid_dbn", "centroid", "Centroid"),
}

# Channels materialised from the ViennaRNA per-sequence cache.  ``bpp``/``stack``
# are stored as probability matrices; ``mfe``/``mea``/``centroid`` are stored as
# dot-bracket strings and converted to a binary pair matrix on load.
THERMO_CHANNELS = ("bpp", "stack", "mfe", "mea", "centroid")
DBN_CHANNELS = ("mfe", "mea", "centroid")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def normalize_sequence(sequence: str) -> str:
    """Upper-case and map DNA thymine to uracil (RiNALMo's own convention)."""
    return sequence.strip().upper().replace("T", "U")


def dbn_to_matrix(dbn: str, length: int) -> np.ndarray:
    """Convert a dot-bracket string to a symmetric binary ``(L, L)`` matrix.

    The thermodynamic cache stores ``mfe_dbn`` / ``mea_dbn`` / ``centroid_dbn``
    as plain ViennaRNA dot-bracket strings.  ``&`` (multi-strand separator) is
    treated as an unpaired position; anything else that is not ``(``, ``)`` or
    ``.`` is a data error rather than something to silently drop.
    """
    dbn = str(dbn)
    length = int(length)
    if len(dbn) != length:
        raise ValueError(f"dot-bracket length {len(dbn)} != sequence length {length}")
    matrix = np.zeros((length, length), dtype=np.float32)
    stack: list[int] = []
    for index, char in enumerate(dbn):
        if char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise ValueError("unbalanced dot-bracket: ')' without matching '('")
            left = stack.pop()
            matrix[left, index] = 1.0
            matrix[index, left] = 1.0
        elif char in ".&":
            continue
        else:
            raise ValueError(f"unsupported dot-bracket character {char!r}")
    if stack:
        raise ValueError("unbalanced dot-bracket: unclosed '('")
    return matrix


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(normalize_sequence(sequence).encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_matrix(
    length: int, pairs: Iterable[Sequence[int]], dtype: np.dtype = np.float32
) -> np.ndarray:
    """Symmetric ``L x L`` 0/1 matrix from 0-based pairs (diagonal stays 0)."""
    matrix = np.zeros((length, length), dtype=dtype)
    for pair in pairs:
        i, j = int(pair[0]), int(pair[1])
        if not (0 <= i < length and 0 <= j < length):
            raise ValueError(f"pair {(i, j)} outside length {length}")
        if i == j:
            raise ValueError(f"self pair {i} is not allowed")
        lo, hi = (i, j) if i < j else (j, i)
        matrix[lo, hi] = 1
        matrix[hi, lo] = 1
    return matrix


def candidate_pairs(sequence: str, min_sep: int = MIN_SEP) -> List[tuple[int, int]]:
    """All eligible candidate pairs (upper triangle, deterministic order).

    Eligibility == prior project's rule: canonical AU/UA/GC/CG/GU/UG and
    ``j - i >= min_sep``.  This is exactly RiNALMo's decode mask (canonical mask
    + sharp-loop mask), so loss/decode/metrics all agree on what is scorable.
    """
    seq = normalize_sequence(sequence)
    length = len(seq)
    pairs: List[tuple[int, int]] = []
    for i in range(length):
        base_i = seq[i]
        for j in range(i + min_sep, length):
            if (base_i, seq[j]) in CANONICAL_PAIRS:
                pairs.append((i, j))
    return pairs


def candidate_mask(sequence: str, min_sep: int = MIN_SEP) -> np.ndarray:
    """Symmetric boolean mask of every eligible candidate pair."""
    length = len(sequence)
    mask = np.zeros((length, length), dtype=bool)
    for i, j in candidate_pairs(sequence, min_sep=min_sep):
        mask[i, j] = True
        mask[j, i] = True
    return mask


def pair_mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """``(B, Lmax, Lmax)`` bool: both endpoints inside the un-padded RNA."""
    if lengths.dim() != 1:
        raise ValueError("lengths must be 1-D")
    max_len = int(lengths.max().item()) if max_len is None else int(max_len)
    valid = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)  # B, Lmax
    return valid.unsqueeze(2) & valid.unsqueeze(1)


# ---------------------------------------------------------------------------
# Tokens / alphabet
# ---------------------------------------------------------------------------
class SimpleAlphabet:
    """Fallback alphabet byte-identical to ``rinalmo.data.alphabet.Alphabet``.

    Used only when the ``rinalmo`` package is unavailable (local CPU tests, CI).
    IDs are the same, so a checkpoint produced with this alphabet is loadable by
    the real RiNALMo tokenizer.
    """

    def __init__(
        self,
        standard_tkns: Sequence[str] = _RNA_TOKENS,
        special_tkns: Sequence[str] = _SPECIAL_TOKENS,
    ):
        self.standard_tkns = list(standard_tkns)
        self.special_tkns = list(special_tkns)
        self.idx_to_tkn = self.special_tkns + self.standard_tkns
        self.tkn_to_idx = {token: index for index, token in enumerate(self.idx_to_tkn)}
        self.cls_idx = self.tkn_to_idx["<cls>"]
        self.eos_idx = self.tkn_to_idx["<eos>"]
        self.unk_idx = self.tkn_to_idx["<unk>"]
        self.pad_idx = self.tkn_to_idx["<pad>"]
        self.mask_idx = self.tkn_to_idx["<mask>"]

    def __len__(self) -> int:
        return len(self.idx_to_tkn)

    def get_tkn(self, index: int) -> str:
        return self.idx_to_tkn[index]

    def get_idx(self, token: str) -> int:
        return self.tkn_to_idx.get(token, self.unk_idx)

    def encode(self, sequence: str, pad_to_len: int = -1) -> List[int]:
        sequence = normalize_sequence(sequence).replace("U", "T")
        encoded = [self.cls_idx] + [self.get_idx(token) for token in sequence] + [self.eos_idx]
        if len(encoded) < pad_to_len:
            encoded = encoded + (pad_to_len - len(encoded)) * [self.pad_idx]
        return encoded


def get_alphabet(prefer_rinalmo: bool = True):
    """Return RiNALMo's Alphabet when importable, else the local fallback."""
    if prefer_rinalmo:
        try:  # lazy: local machines do not ship rinalmo / flash-attn
            from rinalmo.data.alphabet import Alphabet  # type: ignore

            return Alphabet()
        except Exception:
            pass
    return SimpleAlphabet()


def tokenize(
    sequence: str, alphabet=None, pad_to_len: int = -1
) -> torch.Tensor:
    alphabet = alphabet if alphabet is not None else get_alphabet()
    return torch.tensor(alphabet.encode(sequence, pad_to_len=pad_to_len), dtype=torch.long)


# ---------------------------------------------------------------------------
# Canonical JSONL IO
# ---------------------------------------------------------------------------
def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({error})") from error


def validate_record(
    record: Mapping[str, Any],
    *,
    strict_sequence: bool = False,
    strict_pairs: bool = False,
    source: str = "<record>",
) -> dict:
    """Validate one canonical record and return a normalised copy.

    Defaults mirror the **official RiNALMo protocol** (see docs/RESEARCH_PLAN.md
    and the lead's data-contract note):

    * ``strict_sequence=False`` -- IUPAC ambiguity codes are allowed and map to
      ``<unk>`` exactly like the official ``Alphabet``; they are counted and
      disclosed by :meth:`RNAStructureDataset.stats` instead of dropping records
      (5,225 ambiguous bases exist in the official manifest).
    * ``strict_pairs=False`` -- reference dot-bracket pairs are kept verbatim
      even when they involve ambiguous nucleotides or violate canonical/sharp-
      loop rules; those masks apply **only at decode time**.  Range, duplicate
      and single-partner consistency are still enforced because they indicate a
      genuinely corrupt file.

    Set both to ``True`` for a cleaned/filtered variant (e.g. ablations).
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"{source}: record is not a mapping")
    for key in ("id", "sequence", "pairs"):
        if key not in record:
            raise ValueError(f"{source}: missing required key {key!r}")
    identifier = str(record["id"])
    sequence = normalize_sequence(str(record["sequence"]))
    if not sequence:
        raise ValueError(f"{source}/{identifier}: empty sequence")
    if strict_sequence:
        bad = sorted(set(sequence) - set(RNA_ALPHABET))
        if bad:
            raise ValueError(f"{source}/{identifier}: non-ACGU bases {bad}")

    length = int(record.get("length", len(sequence)))
    if length != len(sequence):
        raise ValueError(
            f"{source}/{identifier}: length {length} != len(sequence) {len(sequence)}"
        )

    pairs: List[tuple[int, int]] = []
    for raw in record["pairs"] or []:
        if len(raw) != 2:
            raise ValueError(f"{source}/{identifier}: pair {raw!r} is not a 2-tuple")
        i, j = int(raw[0]), int(raw[1])
        if i > j:
            i, j = j, i
        if i == j:
            raise ValueError(f"{source}/{identifier}: self pair ({i})")
        if not (0 <= i < length and 0 <= j < length):
            raise ValueError(f"{source}/{identifier}: pair ({i}, {j}) out of range")
        if strict_pairs:
            if j - i < MIN_SEP:
                raise ValueError(
                    f"{source}/{identifier}: pair ({i}, {j}) violates min_sep={MIN_SEP}"
                )
            if (sequence[i], sequence[j]) not in CANONICAL_PAIRS:
                raise ValueError(
                    f"{source}/{identifier}: pair ({i}, {j}) "
                    f"{sequence[i]}{sequence[j]} is not canonical"
                )
        pairs.append((i, j))
    if len({p for p in pairs}) != len(pairs):
        raise ValueError(f"{source}/{identifier}: duplicate pairs")
    used = [0] * length
    for i, j in pairs:
        used[i] += 1
        used[j] += 1
        if used[i] > 1 or used[j] > 1:
            raise ValueError(f"{source}/{identifier}: nucleotide paired more than once")

    return {
        "id": identifier,
        "sequence": sequence,
        "pairs": sorted(pairs),
        "length": length,
        "split": str(record.get("split", "")),
        "family": str(record.get("family", record.get("source_id", "unknown"))),
        "source_id": str(record.get("source_id", identifier)),
    }


def load_records(
    paths: Sequence[str | Path],
    *,
    splits: Sequence[str] | None = None,
    limit: int | None = None,
    strict_sequence: bool = False,
    strict_pairs: bool = False,
    dedupe_ids: bool = True,
) -> List[dict]:
    """Read + validate canonical JSONL files, optionally filtering on ``split``."""
    wanted = {str(s) for s in splits} if splits else None
    records: List[dict] = []
    seen: set[str] = set()
    for path in paths:
        for raw in read_jsonl(path):
            record = validate_record(
                raw,
                strict_sequence=strict_sequence,
                strict_pairs=strict_pairs,
                source=str(path),
            )
            if wanted is not None and record["split"] not in wanted:
                continue
            if dedupe_ids:
                if record["id"] in seen:
                    raise ValueError(f"duplicate record id {record['id']!r} in {path}")
                seen.add(record["id"])
            records.append(record)
            if limit is not None and len(records) >= limit:
                return records
    return records


def resolve_dataset_paths(spec: Any, root: str | Path = ".") -> List[Path]:
    """Resolve a config/CLI dataset spec into a sorted list of files.

    Accepted forms::

        "data/datasets/tr0.jsonl.gz"                      # one file
        ["a.jsonl.gz", "b.jsonl.gz"]                      # explicit list
        {"glob": "data/datasets/*.jsonl.gz", "splits": ["TR0"]}
    """
    root = Path(root)
    candidates: List[Path] = []
    if spec is None:
        return []
    if isinstance(spec, (str, Path)):
        item = Path(spec)
        if any(char in str(item) for char in "*?["):
            candidates.extend(Path(p) for p in glob.glob(str(root / item)))
        else:
            candidates.append(item)
    elif isinstance(spec, Mapping):
        pattern = spec.get("glob") or spec.get("path") or spec.get("paths")
        if pattern is None:
            raise ValueError(f"dataset spec {spec!r} has no glob/path/paths")
        if isinstance(pattern, (str, Path)):
            pattern = [pattern]
        for item in pattern:
            item = Path(item)
            if any(char in str(item) for char in "*?["):
                candidates.extend(Path(p) for p in glob.glob(str(root / item)))
            else:
                candidates.append(item)
    elif isinstance(spec, (list, tuple)):
        for item in spec:
            candidates.extend(resolve_dataset_paths(item, root))
    else:
        raise TypeError(f"unsupported dataset spec: {type(spec)!r}")

    resolved: List[Path] = []
    for path in candidates:
        absolute = path if path.is_absolute() else root / path
        if not absolute.exists():
            raise FileNotFoundError(f"dataset file not found: {absolute}")
        resolved.append(absolute)
    return sorted({p.resolve() for p in resolved}, key=str)


def spec_splits(spec: Any) -> Sequence[str] | None:
    """Return the ``splits`` filter declared inside a dataset spec (if any)."""
    if isinstance(spec, Mapping):
        splits = spec.get("splits") or spec.get("split")
        if splits is None:
            return None
        return [splits] if isinstance(splits, str) else list(splits)
    return None


def dataset_fingerprint(spec: Any, root: str | Path = ".") -> dict:
    """sha256 of every file selected by a spec, plus a combined digest."""
    paths = resolve_dataset_paths(spec, root)
    files = {str(path): file_sha256(path) for path in paths}
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"files": files, "combined_sha256": combined, "num_files": len(files)}


# ---------------------------------------------------------------------------
# Thermodynamic feature cache (BPP / stack), keyed by sha256(sequence)
# ---------------------------------------------------------------------------
class ThermoFeatureStore:
    """Read per-sequence ``.npz`` feature caches.

    A cache is one ``.npz`` per unique sequence at
    ``<directory>/<sha256(normalized_sequence)>.npz`` containing ``L x L`` arrays
    for BPP and (optionally) stack probabilities.  Alias key names from the prior
    project are accepted: ``bpp``/``P`` and ``stack``/``Q``.

    ``require_all=True`` turns a missing cache into an error instead of zeros.
    """

    def __init__(
        self,
        directory: str | Path | None,
        channels: Sequence[str] = ("bpp", "stack"),
        *,
        require_all: bool = False,
        clip: bool = True,
    ):
        self.directory = Path(directory) if directory is not None else None
        self.channels = tuple(channels)
        self.require_all = bool(require_all)
        self.clip = bool(clip)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        if self.directory is not None and not self.directory.is_dir():
            raise FileNotFoundError(f"thermo feature dir not found: {self.directory}")

    def __len__(self) -> int:
        if self.directory is None:
            return 0
        return len(list(self.directory.glob("*.npz")))

    def path_for(self, sequence: str) -> Path | None:
        if self.directory is None:
            return None
        return self.directory / f"{sequence_sha256(sequence)}.npz"

    def features(self, sequence: str) -> np.ndarray | None:
        """Return ``(C, L, L)`` float32 channels, or ``None`` when unavailable."""
        if not self.channels:
            return None
        digest = sequence_sha256(sequence)
        if digest in self._cache:
            return self._stack(self._cache[digest], len(sequence)) if self._cache[digest] else None
        path = self.path_for(sequence)
        if path is None or not path.exists():
            if self.require_all:
                raise FileNotFoundError(f"missing thermo cache for sequence {digest}: {path}")
            self._cache[digest] = {}
            return None
        with np.load(path, allow_pickle=False) as payload:
            arrays: Dict[str, np.ndarray] = {}
            for channel in self.channels:
                array = None
                for alias in _CHANNEL_ALIASES.get(channel, (channel,)):
                    if alias in payload:
                        value = payload[alias]
                        if channel in DBN_CHANNELS:
                            if isinstance(value, np.ndarray) and value.shape == ():
                                value = value.item()
                            array = dbn_to_matrix(str(value), len(sequence))
                        else:
                            array = np.asarray(value, dtype=np.float32)
                        break
                if array is None:
                    if self.require_all:
                        raise KeyError(
                            f"{path} has no key for channel {channel!r}; "
                            f"available keys: {list(payload.keys())}"
                        )
                    array = np.zeros((len(sequence), len(sequence)), dtype=np.float32)
                arrays[channel] = array
        self._cache[digest] = arrays
        return self._stack(arrays, len(sequence))

    def _stack(self, arrays: Mapping[str, np.ndarray], length: int) -> np.ndarray:
        channels = []
        for channel in self.channels:
            array = arrays[channel]
            if array.shape != (length, length):
                raise ValueError(
                    f"thermo channel {channel!r} has shape {array.shape}, expected "
                    f"{(length, length)}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"thermo channel {channel!r} contains non-finite values")
            # ViennaRNA's bpp() fills the upper triangle only (lower is all zero);
            # maximum() recovers the symmetric matrix from either storage convention
            # (upper-only or already-symmetrised), while averaging would halve it.
            symmetric = np.maximum(array.astype(np.float32), array.astype(np.float32).T)
            if self.clip:
                symmetric = np.clip(symmetric, 0.0, 1.0)
            np.fill_diagonal(symmetric, 0.0)
            channels.append(symmetric)
        return np.stack(channels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RNAStructureDataset(Dataset):
    """Canonical records -> tensors.

    ``__getitem__`` returns a dict of *unpadded* tensors::

        id, sequence, family, split, pairs
        tokens   LongTensor[L+2]      # <cls> seq <eos>
        length   int
        target   Float32Tensor[L, L]  # symmetric 0/1 reference pairs
        eligible BoolTensor[L, L]     # canonical + min_sep candidate mask
        channels Float32Tensor[C, L, L]  # only when ``channels`` is non-empty

    ``channels`` order is fixed by ``self.channels``; supported data-level
    channels: ``bpp``, ``stack``, ``mfe``, ``mea``, ``centroid``, ``distance``,
    ``mask`` (``s0`` is produced inside the pair head, never by the dataset).
    """

    DATA_CHANNELS = ("bpp", "stack", "mfe", "mea", "centroid", "distance", "mask")

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        alphabet=None,
        channels: Sequence[str] = (),
        thermo: ThermoFeatureStore | None = None,
        min_sep: int = MIN_SEP,
        max_length: int | None = None,
        require_eligible: bool = False,
        cache_arrays: bool = True,
    ):
        if isinstance(records, (str, Path)):
            raise TypeError("pass records (see load_records), not a path")
        unknown = [c for c in channels if c not in self.DATA_CHANNELS]
        if unknown:
            raise ValueError(f"unknown data channels {unknown}; supported {self.DATA_CHANNELS}")
        if channels and any(c in THERMO_CHANNELS for c in channels) and thermo is None:
            raise ValueError("thermodynamic channels requested but no ThermoFeatureStore was given")
        self.records = list(records)
        self.alphabet = alphabet if alphabet is not None else get_alphabet()
        self.channels = tuple(channels)
        self.thermo = thermo
        self.min_sep = int(min_sep)
        self.max_length = max_length
        self.require_eligible = bool(require_eligible)
        self._arrays: List[dict | None] = [None] * len(self.records)
        self._cache_arrays = bool(cache_arrays)
        if not self.records:
            raise ValueError("RNAStructureDataset received zero records")

    # -- introspection ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    @property
    def lengths(self) -> List[int]:
        return [int(r["length"]) for r in self.records]

    @property
    def families(self) -> List[str]:
        return [str(r["family"]) for r in self.records]

    def split_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record["split"]] = counts.get(record["split"], 0) + 1
        return counts

    def stats(self) -> Dict[str, Any]:
        """Disclosed dataset statistics (ambiguity / ineligible reference pairs).

        The official protocol keeps ambiguous bases and non-canonical reference
        pairs; dropping or silently rewriting them would change the benchmark.
        These counters make the deviation measurable instead.
        """
        ambiguous_bases = 0
        records_with_ambiguity = 0
        ineligible_reference_pairs = 0
        total_reference_pairs = 0
        max_length = 0
        for record in self.records:
            sequence = record["sequence"]
            ambiguous = sum(1 for base in sequence if base not in RNA_ALPHABET)
            ambiguous_bases += ambiguous
            records_with_ambiguity += int(ambiguous > 0)
            length = int(record["length"])
            max_length = max(max_length, length)
            eligible = candidate_mask(sequence, min_sep=self.min_sep)
            for i, j in record["pairs"]:
                total_reference_pairs += 1
                if not eligible[i, j]:
                    ineligible_reference_pairs += 1
        return {
            "num_records": len(self.records),
            "splits": self.split_counts(),
            "ambiguous_bases": ambiguous_bases,
            "records_with_ambiguity": records_with_ambiguity,
            "unknown_token_records": records_with_ambiguity,
            "total_reference_pairs": total_reference_pairs,
            "ineligible_reference_pairs": ineligible_reference_pairs,
            "ineligible_reference_fraction": (
                ineligible_reference_pairs / total_reference_pairs
                if total_reference_pairs
                else 0.0
            ),
            "max_length": max_length,
            "min_length": min(int(r["length"]) for r in self.records),
        }

    def filter_by_length(self, max_length: int) -> "RNAStructureDataset":
        keep = [r for r in self.records if int(r["length"]) <= int(max_length)]
        if not keep:
            raise ValueError(f"no records with length <= {max_length}")
        filtered = RNAStructureDataset(
            keep,
            alphabet=self.alphabet,
            channels=self.channels,
            thermo=self.thermo,
            min_sep=self.min_sep,
            max_length=max_length,
            require_eligible=self.require_eligible,
            cache_arrays=self._cache_arrays,
        )
        report = getattr(self, "filter_report", None)
        if report is not None:
            filtered.filter_report = report
        return filtered

    # -- item construction -------------------------------------------------
    def _arrays_for(self, index: int) -> dict:
        cached = self._arrays[index] if self._cache_arrays else None
        if cached is not None:
            return cached
        record = self.records[index]
        sequence = record["sequence"]
        length = int(record["length"])
        target = pair_matrix(length, record["pairs"], dtype=np.float32)
        eligible = candidate_mask(sequence, min_sep=self.min_sep)
        if self.require_eligible:
            uncovered = target.astype(bool) & ~eligible
            if uncovered.any():
                i, j = np.argwhere(uncovered)[0]
                raise ValueError(
                    f"{record['id']}: reference pair ({i}, {j}) is not eligible "
                    f"(canonical? min_sep={self.min_sep}?)"
                )
        arrays = {
            "target": target,
            "eligible": eligible,
            "tokens": torch.tensor(
                self.alphabet.encode(sequence), dtype=torch.long
            ),
        }
        if self.channels:
            channel_arrays = []
            for channel in self.channels:
                if channel in THERMO_CHANNELS:
                    features = self.thermo.features(sequence)  # type: ignore[union-attr]
                    if features is None:
                        raise FileNotFoundError(
                            f"{record['id']}: missing thermo cache for channel {channel!r}"
                        )
                    channel_arrays.append(features[self.channels.index(channel)])
                elif channel == "distance":
                    positions = np.arange(length, dtype=np.float32)
                    distance = np.abs(positions[:, None] - positions[None, :])
                    channel_arrays.append(distance / max(length - 1, 1))
                elif channel == "mask":
                    channel_arrays.append(eligible.astype(np.float32))
            arrays["channels"] = np.stack(channel_arrays).astype(np.float32)
        if self._cache_arrays:
            self._arrays[index] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        arrays = self._arrays_for(index)
        item = {
            "id": record["id"],
            "sequence": record["sequence"],
            "family": record["family"],
            "split": record["split"],
            "pairs": record["pairs"],
            "length": int(record["length"]),
            "tokens": arrays["tokens"],
            "target": torch.from_numpy(arrays["target"]),
            "eligible": torch.from_numpy(arrays["eligible"]),
        }
        if self.channels:
            item["channels"] = torch.from_numpy(arrays["channels"])
        return item


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def collate_batch(batch: Sequence[Mapping[str, Any]]) -> dict:
    """Pad a list of dataset items into a training/eval batch.

    Returns tensors plus the metadata needed for decoding/metrics:

    ================  =====================================================
    tokens            LongTensor  (B, T=max_len+2)  <cls> seq <eos> pad...
    padding_mask      BoolTensor  (B, T)  True == padding (torch convention)
    lengths           LongTensor  (B,)
    target            FloatTensor (B, Lmax, Lmax)
    eligible          BoolTensor  (B, Lmax, Lmax)
    pair_mask         BoolTensor  (B, Lmax, Lmax)  both endpoints < length
    channels          FloatTensor (B, C, Lmax, Lmax)  only if items carry it
    ids/sequences/families/pairs   python lists (un-padded ground truth)
    ================  =====================================================
    """
    if not batch:
        raise ValueError("empty batch")
    batch_size = len(batch)
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    max_tokens = max_len + 2  # <cls> + seq + <eos>

    tokens = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    target = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
    eligible = torch.zeros((batch_size, max_len, max_len), dtype=torch.bool)

    has_channels = "channels" in batch[0]
    if has_channels:
        channel_count = int(batch[0]["channels"].shape[0])
        channels = torch.zeros(
            (batch_size, channel_count, max_len, max_len), dtype=torch.float32
        )
    else:
        channels = None

    for row, item in enumerate(batch):
        length = int(item["length"])
        item_tokens = item["tokens"]
        if item_tokens.numel() != length + 2:
            raise ValueError(
                f"{item['id']}: token length {item_tokens.numel()} != length+2 ({length + 2})"
            )
        tokens[row, : length + 2] = item_tokens
        target[row, :length, :length] = item["target"]
        eligible[row, :length, :length] = item["eligible"]
        if has_channels:
            channels[row, :, :length, :length] = item["channels"]

    # Default pad index matches RiNALMo's Alphabet (<pad> == 1).  We only need
    # padding_mask downstream, so the token value is irrelevant.
    padding_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool)
    padding_mask[:, : max_len + 2] = False
    for row, item in enumerate(batch):
        padding_mask[row, : int(item["length"]) + 2] = False

    batch_out = {
        "tokens": tokens,
        "padding_mask": padding_mask,
        "lengths": lengths,
        "target": target,
        "eligible": eligible,
        "pair_mask": pair_mask_from_lengths(lengths, max_len),
        "ids": [item["id"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "families": [item["family"] for item in batch],
        "pairs": [item["pairs"] for item in batch],
    }
    if channels is not None:
        batch_out["channels"] = channels
    return batch_out


# ---------------------------------------------------------------------------
# Length-bucketed batching
# ---------------------------------------------------------------------------
class BucketBatchSampler(Sampler[List[int]]):
    """Length-sorted buckets -> batches, to bound padding.

    Sorting is global (lengths), buckets are contiguous chunks of ``bucket_size``
    and batches never cross bucket boundaries unless the remainder is merged
    with the previous bucket (which keeps every sample exactly once).
    ``set_epoch`` reshuffles within buckets and bucket order deterministically.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        bucket_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        max_tokens: int | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if bucket_size < batch_size:
            bucket_size = batch_size
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.epoch = 0

    def _bucket_batch_size(self, bucket: Sequence[int]) -> int:
        """Batch size for one length-sorted bucket, capped by the token budget."""
        if not bucket:
            return self.batch_size
        if self.max_tokens:
            max_len = int(self.lengths[max(bucket)])
            return max(1, min(self.batch_size, self.max_tokens // max(1, max_len)))
        return self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _pack(self) -> List[List[int]]:
        """Greedy length-sorted packing under the batch-size and token budget.

        Sequences are processed in ascending length order; a batch grows while it
        stays within ``batch_size`` and (when set) within ``max_tokens`` tokens of
        padding-free work, i.e. ``len(batch) * max_length_in_batch <= max_tokens``.
        This is what keeps a 24 GB card safe when a few very long RNAs appear.
        """
        order = np.argsort(self.lengths, kind="stable").tolist()
        batches: List[List[int]] = []
        current: List[int] = []
        for index in order:
            length = int(self.lengths[index])
            if current:
                new_len = max(int(self.lengths[current[-1]]), length)
                too_many = len(current) >= self.batch_size
                too_long = bool(self.max_tokens) and (len(current) + 1) * new_len > self.max_tokens
                if too_many or too_long:
                    batches.append(current)
                    current = [index]
                    continue
            current.append(index)
        if current:
            batches.append(current)
        if self.drop_last and batches:
            batches = [b for b in batches if len(b) == self.batch_size]
        return batches

    def __len__(self) -> int:
        return len(self._pack())

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        batches = self._pack()
        if self.shuffle:
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
        for batch in batches:
            yield batch


def make_dataloader(
    dataset: RNAStructureDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    bucket_size: int = 64,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 0,
    max_tokens: int | None = None,
) -> torch.utils.data.DataLoader:
    sampler = BucketBatchSampler(
        dataset.lengths,
        batch_size,
        bucket_size=bucket_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        max_tokens=max_tokens,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_batch,
        num_workers=int(num_workers),
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Sequence-level split audit helpers (used by train/evaluate provenance)
# ---------------------------------------------------------------------------
def sequence_overlap(left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]) -> int:
    """Count exact normalized-sequence overlap between two record collections."""
    left_hashes = {sequence_sha256(r["sequence"]) for r in left}
    right_hashes = {sequence_sha256(r["sequence"]) for r in right}
    return len(left_hashes & right_hashes)


def read_ct_file(path: str | Path) -> dict:
    """Parse one CT file (1-based on disk) into a canonical 0-based record.

    CT columns: ``index base index-1 index+1 partner natural-numbering``.
    Used by ``scripts/eval_official.py`` to score the official ArchiveII
    ``ct-splits`` trees directly, without waiting for a JSONL conversion.
    """
    path = Path(path)
    rows = [
        row.split()
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    if not rows:
        raise ValueError(f"{path}: empty CT file")
    length = int(rows[0][0])
    if len(rows) != length + 1:
        raise ValueError(f"{path}: header length {length} but {len(rows) - 1} rows")
    sequence = "".join(row[1] for row in rows[1:])
    partners: Dict[int, int] = {}
    for row in rows[1:]:
        if len(row) < 5:
            raise ValueError(f"{path}: malformed CT row {row!r}")
        index = int(row[0]) - 1
        partner = int(row[4]) - 1
        if partner >= 0:
            partners[index] = partner
    for index, partner in partners.items():
        if partners.get(partner) != index:
            raise ValueError(f"{path}: asymmetric partner at {index}->{partner}")
    pairs = sorted((i, j) for i, j in partners.items() if i < j)
    return validate_record(
        {
            "id": path.stem,
            "sequence": sequence,
            "pairs": pairs,
            "length": length,
            "split": path.parent.name,
            "family": path.parent.parent.name if path.parent.parent.name else "unknown",
            "source_id": str(path),
        },
        strict_sequence=False,
        strict_pairs=False,
        source=str(path),
    )


def load_ct_records(
    path: str | Path,
    *,
    family: str | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> List[dict]:
    """Load every ``*.ct`` file under ``path`` (recursively) as canonical records."""
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.ct"), key=str)
    else:
        raise FileNotFoundError(f"CT path not found: {path}")
    records = []
    for file_path in files:
        record = read_ct_file(file_path)
        if family is not None:
            record["family"] = str(family)
        if split is not None:
            record["split"] = str(split)
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"no CT files found under {path}")
    return records


def build_dataset_for_spec(
    spec: Any,
    *,
    root: str | Path = ".",
    channels: Sequence[str] = (),
    thermo: ThermoFeatureStore | None = None,
    alphabet=None,
    limit: int | None = None,
    min_sep: int = MIN_SEP,
    max_length: int | None = None,
    strict_sequence: bool = False,
    strict_pairs: bool = False,
    require_eligible: bool = False,
) -> "RNAStructureDataset":
    """Resolve a spec, load + validate records, build one dataset (shared by train/eval)."""
    paths = resolve_dataset_paths(spec, root)
    records = load_records(
        paths,
        splits=spec_splits(spec),
        limit=limit,
        strict_sequence=strict_sequence,
        strict_pairs=strict_pairs,
    )
    if not records:
        raise RuntimeError(f"dataset spec {spec!r} produced zero records")
    dataset = RNAStructureDataset(
        records,
        alphabet=alphabet,
        channels=channels,
        thermo=thermo,
        min_sep=min_sep,
        max_length=max_length,
        require_eligible=require_eligible,
    )
    if max_length is not None:
        over = [record["id"] for record in dataset.records if int(record["length"]) > int(max_length)]
        if over:
            dataset.filter_report = {
                "max_length": int(max_length),
                "num_dropped": len(over),
                "dropped_ids": over[:100],
                "dropped_ids_truncated": len(over) > 100,
            }
            dataset = dataset.filter_by_length(int(max_length))
    return dataset


def dataset_summary(records: Sequence[Mapping[str, Any]]) -> dict:
    lengths = np.asarray([int(r["length"]) for r in records], dtype=np.int64)
    pair_counts = np.asarray([len(r["pairs"]) for r in records], dtype=np.int64)
    families: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    for record in records:
        families[record["family"]] = families.get(record["family"], 0) + 1
        splits[record["split"]] = splits.get(record["split"], 0) + 1
    return {
        "num_records": len(records),
        "splits": splits,
        "families": dict(sorted(families.items())),
        "length": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
        },
        "pairs": {
            "total": int(pair_counts.sum()),
            "mean": float(pair_counts.mean()),
            "max": int(pair_counts.max()),
        },
    }
