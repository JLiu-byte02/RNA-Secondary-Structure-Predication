#!/usr/bin/env python
"""Build the authoritative local RNA secondary-structure datasets.

One-command rebuild of every canonical artifact under ``data/``:

    "/e/python 3.12.9/python.exe" scripts/build_datasets.py

Sources (read-only):
  * ../data/processed/manifest.tsv.gz + split_ids.json  (prior-project export of bpRNA)
  * data/raw/ct-splits.tar.gz                            (official dl-rna ArchiveII release)

Outputs:
  data/datasets/official/{TR0,VL0,TS0,bprna_new}.jsonl.gz         (all 18,820 manifest rows,
      grouped by ``original_split`` exactly as shipped; NO filtering)
  data/datasets/canonical_filtered/{TR0,VL0,TS0,bprna_new,
      fit_train,family_dev}.jsonl.gz                              (prior-project split_ids.json
      membership; the 176 non-ACGU rows are excluded here by definition of that split)
  data/datasets/archiveii/<family>.<role>.jsonl.gz                (9 official family folds x
      train/valid/test, membership byte-identical to the official ct-splits)
  data/datasets/smoke8.jsonl.gz                                   (8 synthetic RNAs, CPU smoke tests)
  data/metadata/archiveii_records.json.gz                         (per-CT-file provenance + hashes)
  data/integrity_report.json                                      (hashes, counts, audits, checks)
  data/DATASET_CARD.md                                            (human-readable card)

Record schema (every JSONL file, 0-based inclusive pairs, i < j, sorted):
    {"id": str, "sequence": str, "pairs": [[i0, j0], ...], "length": int,
     "split": str, "family": str|null, "source_id": str}
ArchiveII records additionally carry ``"fold": "<held-out family>"``.

Design rules
  * Nothing is filtered, deduplicated or "fixed": anomalies are reported in the
    integrity report and the dataset card.
  * All gzip members are written with mtime=0 and the JSONL stream is fully
    sorted, so a rebuild is byte-reproducible on the same Python/zlib.
  * Fail loudly: every invariant is either an exit-2 hard error or a ``checks``
    entry; ``passed`` is true only when every check passes.

``--check`` re-verifies the on-disk outputs against ``integrity_report.json``
without rewriting anything.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import io
import json
import math
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- constants
FAMILIES = ("5s", "16s", "23s", "tRNA", "srp", "tmRNA", "RNaseP", "grp1", "telomerase")
ROLES = ("train", "valid", "test")

ARCHIVE_URL = "https://github.com/marcellszi/dl-rna/releases/download/Data/ct-splits.tar.gz"
ARCHIVE_SHA256 = "0ff7209016ba1775288572794e850d5857937c391d67c9ccacfe9ae77c0188e5"
ARCHIVE_SIZE = 33_015_660
# The task text quotes 23s:315 in one place but also "23s:15 per original"; the
# official release is 15 (total 3,865). We assert the official total and the
# per-family counts below, and report the 23s discrepancy in the report.
ARCHIVE_TEST_EXPECTED = {
    "5s": 1283, "16s": 67, "23s": 15, "tRNA": 557, "srp": 918,
    "tmRNA": 462, "RNaseP": 454, "grp1": 74, "telomerase": 35,
}
ARCHIVE_TEST_TOTAL = 3865
ARCHIVE_TASK_TEXT_23S = 315  # disclosed, not used as an assertion

OFFICIAL_SPLITS = ("TR0", "VL0", "TS0", "bprna_new")
OFFICIAL_COUNTS = {"TR0": 10814, "VL0": 1300, "TS0": 1305, "bprna_new": 5401}
# Lead-specified filenames: data/datasets/official_bprna_{train,valid,test,bprna_new}.jsonl.gz
OFFICIAL_FILE_STEM = {"TR0": "official_bprna_train", "VL0": "official_bprna_valid",
                      "TS0": "official_bprna_test", "bprna_new": "official_bprna_bprna_new"}
FILTERED_SPLITS = ("TR0", "VL0", "TS0", "bprna_new", "fit_train", "family_dev")
FILTERED_COUNTS = {"TR0": 10682, "VL0": 1286, "TS0": 1288, "bprna_new": 5388,
                   "fit_train": 3299, "family_dev": 1068}
MANIFEST_ROWS = 18820
EXCLUDED_ROWS = 176
KMER_K = 8
CODE = {"A": 0, "C": 1, "G": 2, "U": 3}
PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

WS = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = WS.parent / "data" / "processed" / "manifest.tsv.gz"
DEFAULT_SPLIT_IDS = WS.parent / "data" / "processed" / "split_ids.json"
DEFAULT_ARCHIVE = WS / "data" / "raw" / "ct-splits.tar.gz"


class BuildError(RuntimeError):
    """Raised when a hard invariant fails; the build aborts (exit 2)."""


# --------------------------------------------------------------------------- helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def membership_sha(ids) -> str:
    return sha256_text("\n".join(sorted(ids)) + "\n")


def round6(x):
    return None if x is None else round(float(x), 6)


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return int(round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac))


def dist(values):
    vals = sorted(values)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals), "min": vals[0], "max": vals[-1],
        "mean": round6(sum(vals) / len(vals)),
        "median": pct(vals, 0.50), "p90": pct(vals, 0.90),
        "p95": pct(vals, 0.95), "p99": pct(vals, 0.99),
    }


def write_json(path: Path, value) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def write_jsonl_gz(path: Path, records: list[dict]) -> dict:
    """Deterministic gzipped JSONL. Returns hashes/sizes for the report."""
    lines = "".join(
        json.dumps(r, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"
        for r in records
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=6) as gz:
        gz.write(lines)
    payload = buf.getvalue()
    path.write_bytes(payload)
    return {
        "sha256": sha256_bytes(payload),
        "content_sha256": sha256_bytes(lines),
        "bytes": len(payload),
        "uncompressed_bytes": len(lines),
        "records": len(records),
    }


def normalize_sequence(seq: str) -> str:
    """Uppercase + T->U, used only for sequence-identity audits (never to rewrite data)."""
    return seq.upper().replace("T", "U")


def encode_kmers(seq: str, k: int = KMER_K) -> set[int]:
    """2-bit encoded set of distinct ACGU-only k-mers; ambiguous windows are skipped."""
    out: set[int] = set()
    v = 0
    run = 0
    mask = (1 << (2 * k)) - 1
    for ch in seq:
        c = CODE.get(ch)
        if c is None:
            v = 0
            run = 0
            continue
        v = ((v << 2) | c) & mask
        run += 1
        if run >= k:
            out.add(v)
    return out


def encode_kmers_multi(seq: str, ks) -> dict[int, set[int]]:
    """One pass over ``seq`` yielding distinct ACGU-only k-mer sets for several k."""
    ks = tuple(sorted(set(ks)))
    out = {k: set() for k in ks}
    masks = {k: (1 << (2 * k)) - 1 for k in ks}
    max_mask = masks[ks[-1]]
    v = 0
    run = 0
    for ch in seq:
        c = CODE.get(ch)
        if c is None:
            v = 0
            run = 0
            continue
        v = ((v << 2) | c) & max_mask
        run += 1
        for k in ks:
            if run >= k:
                out[k].add(v & masks[k])
    return out


def sequence_alphabet(sequences) -> dict:
    counter = Counter()
    for s in sequences:
        counter.update(s)
    return dict(sorted(counter.items()))


def parse_pairs_1based(text: str, length: int, context: str):
    """Parse 'i-j;...' (1-based inclusive) into validated 0-based sorted pairs."""
    pairs = []
    if text:
        for token in text.split(";"):
            if not token:
                continue
            try:
                a_s, b_s = token.split("-")
                a, b = int(a_s), int(b_s)
            except ValueError as exc:
                raise BuildError(f"{context}: malformed pair token {token!r}") from exc
            if not (1 <= a <= length and 1 <= b <= length):
                raise BuildError(f"{context}: pair {token!r} outside 1..{length}")
            if a == b:
                raise BuildError(f"{context}: self pair {token!r}")
            if a > b:  # manifest is shipped i<j; a reversed token is a real anomaly
                raise BuildError(f"{context}: pair {token!r} not ordered i<j")
            pairs.append((a - 1, b - 1))
    if len(set(pairs)) != len(pairs):
        raise BuildError(f"{context}: duplicate pairs")
    used = set()
    for i, j in pairs:
        if i in used or j in used:
            raise BuildError(f"{context}: position paired more than once")
        used.add(i)
        used.add(j)
    return sorted(pairs)


def crossing_stats(pairs) -> int:
    """Number of crossing pair-pairs (pseudoknot event count) for one RNA."""
    js = []
    crossings = 0
    for i, j in sorted(pairs):
        # previous pairs all have i' < i; they cross iff i < j' < j
        lo = bisect.bisect_right(js, i)
        hi = bisect.bisect_left(js, j)
        if hi > lo:
            crossings += hi - lo
        bisect.insort(js, j)
    return crossings


def parse_manifest(path: Path):
    if not path.exists():
        raise BuildError(f"manifest not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        required = {"sample_id", "source_id", "sequence", "sequence_hash", "length",
                    "original_pairs", "eligible_pairs", "original_split",
                    "working_split", "family_id", "excluded_reason"}
        missing = required - set(header)
        if missing:
            raise BuildError(f"manifest missing columns: {sorted(missing)}")
        rows = []
        for line_no, line in enumerate(f, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                raise BuildError(f"manifest line {line_no}: {len(fields)} fields != {len(header)}")
            rows.append(dict(zip(header, fields)))
    return header, rows


def validate_manifest_rows(rows):
    if len(rows) != MANIFEST_ROWS:
        raise BuildError(f"manifest row count {len(rows)} != expected {MANIFEST_ROWS}")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise BuildError("duplicate sample_id in manifest")
    source_ids = [r["source_id"] for r in rows]
    if len(set(source_ids)) != len(source_ids):
        raise BuildError("duplicate source_id in manifest")
    bad_hash = [r["sample_id"] for r in rows
                if sha256_text(r["sequence"]) != r["sequence_hash"]]
    if bad_hash:
        raise BuildError(f"sequence_hash mismatch for {len(bad_hash)} rows, e.g. {bad_hash[:3]}")
    bad_len = [r["sample_id"] for r in rows if int(r["length"]) != len(r["sequence"])]
    if bad_len:
        raise BuildError(f"length mismatch for {len(bad_len)} rows, e.g. {bad_len[:3]}")
    for r in rows:  # pair syntax/range validated for every row, filtered or not
        parse_pairs_1based(r["original_pairs"], len(r["sequence"]), r["sample_id"])
        parse_pairs_1based(r["eligible_pairs"], len(r["sequence"]), r["sample_id"] + "/eligible")


def load_split_ids(path: Path):
    if not path.exists():
        raise BuildError(f"split_ids.json not found: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != set(FILTERED_SPLITS):
        raise BuildError(f"split_ids keys {sorted(payload)} != {sorted(FILTERED_SPLITS)}")
    out = {}
    for name in FILTERED_SPLITS:
        ids = payload[name]
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise BuildError(f"split_ids[{name!r}] is not a list of strings")
        if len(set(ids)) != len(ids):
            raise BuildError(f"split_ids[{name!r}] contains duplicates")
        out[name] = ids
    return out


# --------------------------------------------------------------------------- bpRNA build
def bprna_record(row, split: str, include_fold: bool = False) -> dict:
    seq = row["sequence"]
    pairs = parse_pairs_1based(row["original_pairs"], len(seq), row["sample_id"])
    rec = {
        "id": row["sample_id"],
        "sequence": seq,
        "pairs": [[i, j] for i, j in pairs],
        "length": len(seq),
        "split": split,
        "family": row["family_id"] or None,
        "source_id": row["source_id"],
    }
    if include_fold:
        rec["fold"] = split
    return rec


def build_bprna_family(rows, groups, out_dir: Path, kind: str, expected_counts,
                       filename_fn=None):
    filename_fn = filename_fn or (lambda split: f"{split}.jsonl.gz")
    summaries = {}
    for split in groups:
        ids = groups[split]
        if len(ids) != expected_counts[split]:
            raise BuildError(f"{kind}/{split}: {len(ids)} records != expected {expected_counts[split]}")
        recs = [bprna_record(rows[i], split) for i in ids]
        recs.sort(key=lambda r: r["id"])
        if len({r["id"] for r in recs}) != len(recs):
            raise BuildError(f"{kind}/{split}: duplicate ids after build")
        path = out_dir / filename_fn(split)
        file_info = write_jsonl_gz(path, recs)
        summaries[split] = {
            "records": len(recs),
            "ids_sha256": membership_sha([r["id"] for r in recs]),
            "length": dist([r["length"] for r in recs]),
            "pairs": pair_summary(recs),
            "alphabet": sequence_alphabet([r["sequence"] for r in recs]),
            "family_label_present": sum(1 for r in recs if r["family"]),
            "family_labels": len({r["family"] for r in recs if r["family"]}),
            "file": relative_path(path),
            **file_info,
        }
    return summaries


def pair_summary(records) -> dict:
    counts = [len(r["pairs"]) for r in records]
    crossings = [crossing_stats([tuple(p) for p in r["pairs"]]) for r in records]
    paired_positions = sum(2 * c for c in counts)
    total_positions = sum(r["length"] for r in records)
    return {
        "total_pairs": sum(counts),
        "empty_pair_rnas": sum(1 for c in counts if c == 0),
        "per_rna": dist(counts),
        "paired_fraction": round6(paired_positions / total_positions) if total_positions else None,
        "pseudoknot": {
            "rnas_with_crossing_pairs": sum(1 for c in crossings if c > 0),
            "crossing_pair_pairs": sum(crossings),
        },
    }


def relative_path(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(WS.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return "../" + p.relative_to(WS.parent.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


# --------------------------------------------------------------------------- ArchiveII
def read_ct_bytes(data: bytes, name: str):
    text = data.decode("utf-8", "replace")
    rows = [r.split() for r in text.splitlines() if r.strip() and not r.lstrip().startswith("#")]
    if not rows:
        raise BuildError(f"{name}: empty CT file")
    try:
        length = int(rows[0][0])
    except (ValueError, IndexError) as exc:
        raise BuildError(f"{name}: bad length header {rows[0]!r}") from exc
    body = rows[1:]
    if len(body) != length:
        raise BuildError(f"{name}: {len(body)} residue lines != declared length {length}")
    sequence = "".join(r[1] for r in body)
    if len(sequence) != length:
        raise BuildError(f"{name}: sequence length {len(sequence)} != {length}")
    partners = {}
    for r in body:
        if len(r) < 5:
            raise BuildError(f"{name}: CT line with <5 columns: {r!r}")
        idx = int(r[0])
        partner = int(r[4])  # CT columns: index base prev next pair [numbering]
        if partner == 0:
            continue
        if not (1 <= idx <= length and 1 <= partner <= length):
            raise BuildError(f"{name}: partner {partner} for {idx} outside 1..{length}")
        if partner in partners and partners[partner] != idx:
            raise BuildError(f"{name}: inconsistent partner for position {partner}")
        partners[partner] = idx
    pairs = []
    for i, j in partners.items():
        if partners.get(j) != i:
            raise BuildError(f"{name}: asymmetric pair {i}-{j}")
        if i < j:
            pairs.append((i - 1, j - 1))
    if len(set(pairs)) != len(pairs):
        raise BuildError(f"{name}: duplicate pairs")
    if len(set(sum(([i, j] for i, j in pairs), []))) != 2 * len(pairs):
        raise BuildError(f"{name}: position paired more than once")
    return sequence, sorted(pairs)


def build_archiveii(archive: Path, out_dir: Path, meta_dir: Path):
    if not archive.exists():
        raise BuildError(
            f"ArchiveII archive missing: {archive}\n"
            "Local network is blocked; fetch it with data/raw/fetch_ct_server.sh on the compute "
            "server and copy back, then re-run. Expected sha256 " + ARCHIVE_SHA256)
    actual_sha = sha256_file(archive)
    if actual_sha != ARCHIVE_SHA256:
        raise BuildError(f"ArchiveII sha256 {actual_sha} != pinned {ARCHIVE_SHA256}")
    archive_size = archive.stat().st_size

    entries = defaultdict(list)  # (family, role) -> [(id, path, bytes)]
    skipped_kfold = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            parts = Path(member.name).parts
            if parts[:2] == ("ct", "k-fold"):
                skipped_kfold += 1  # 5-fold random CV tree in the same release: out of scope
                continue
            if len(parts) < 5 or parts[0] != "ct" or parts[1] != "fam-fold":
                continue
            if parts[2] not in FAMILIES or parts[3] not in ROLES or not member.name.endswith(".ct"):
                raise BuildError(f"unexpected archive member: {member.name}")
            if not member.isfile() or member.size > 2_000_000:
                raise BuildError(f"unexpected archive member type/size: {member.name}")
            data = tar.extractfile(member).read()
            entries[(parts[2], parts[3])].append((Path(member.name).stem, member.name, data))

    parsed = {}  # id -> dict(family, sequence, pairs, ct_sha256, ct_path)
    fold_roles = {}
    for family in FAMILIES:
        for role in ROLES:
            items = entries.get((family, role), [])
            if not items:
                raise BuildError(f"ArchiveII fold {family}/{role}: no CT files")
            fold_roles[(family, role)] = items
            for identifier, path, data in items:
                sequence, pairs = read_ct_bytes(data, path)
                digest = sha256_bytes(data)
                if identifier in parsed:
                    if parsed[identifier]["ct_sha256"] != digest:
                        raise BuildError(f"{identifier}: CT bytes differ across fold copies ({path})")
                    if parsed[identifier]["sequence"] != sequence or \
                            parsed[identifier]["pairs"] != pairs:
                        raise BuildError(f"{identifier}: parsed CT differs across fold copies ({path})")
                    continue
                parsed[identifier] = {
                    "family": None,
                    "sequence": sequence,
                    "pairs": pairs,
                    "ct_sha256": digest,
                    "ct_path": path,
                }
    # The held-out family of a record is defined by the test folder it appears in.
    for family in FAMILIES:
        for identifier, _, _ in fold_roles[(family, "test")]:
            record = parsed[identifier]
            if record["family"] is not None and record["family"] != family:
                raise BuildError(f"{identifier}: in test folders of both {record['family']} and {family}")
            record["family"] = family
    # every record must be observed in a test folder (its own family)
    missing_family = [i for i, r in parsed.items() if r["family"] is None]
    if missing_family:
        raise BuildError(f"{len(missing_family)} CT ids never appear in a test folder")
    record_families = Counter(r["family"] for r in parsed.values())
    test_counts = {f: record_families.get(f, 0) for f in FAMILIES}
    for family in FAMILIES:
        if test_counts[family] != ARCHIVE_TEST_EXPECTED[family]:
            raise BuildError(
                f"ArchiveII {family}: test count {test_counts[family]} != official "
                f"{ARCHIVE_TEST_EXPECTED[family]}")
    if len(parsed) != ARCHIVE_TEST_TOTAL:
        raise BuildError(f"ArchiveII test records {len(parsed)} != {ARCHIVE_TEST_TOTAL}")

    all_ids = set(parsed)
    fold_summaries = {}
    files = {}
    for family in FAMILIES:
        roles = {}
        for role in ROLES:
            ids = [identifier for identifier, _, _ in fold_roles[(family, role)]]
            if len(set(ids)) != len(ids):
                raise BuildError(f"ArchiveII {family}/{role}: duplicate ids")
            roles[role] = ids
        if set().union(*[set(v) for v in roles.values()]) != all_ids:
            raise BuildError(f"ArchiveII fold {family}: roles do not cover all {len(all_ids)} records")
        total = sum(len(v) for v in roles.values())
        if total != len(all_ids):
            raise BuildError(f"ArchiveII fold {family}: roles overlap (sum {total} != {len(all_ids)})")
        leaked = [i for i in roles["test"] if parsed[i]["family"] != family]
        if leaked:
            raise BuildError(f"ArchiveII fold {family}: {len(leaked)} test ids not in held-out family")
        misassigned = [i for role in ("train", "valid") for i in roles[role]
                       if parsed[i]["family"] == family]
        if misassigned:
            raise BuildError(
                f"ArchiveII fold {family}: {len(misassigned)} held-out-family ids leak into "
                f"train/valid")
        role_counts = {}
        for role in ROLES:
            recs = []
            for identifier in roles[role]:
                p = parsed[identifier]
                recs.append({
                    "id": identifier,
                    "sequence": p["sequence"],
                    "pairs": [[i, j] for i, j in p["pairs"]],
                    "length": len(p["sequence"]),
                    "split": role,
                    "family": p["family"],
                    "source_id": identifier,
                    "fold": family,
                })
            recs.sort(key=lambda r: r["id"])
            path = out_dir / family / f"{role}.jsonl.gz"
            info = write_jsonl_gz(path, recs)
            files[f"{family}.{role}"] = {"path": relative_path(path), **info}
            role_counts[role] = {
                "records": len(recs),
                "ids_sha256": membership_sha([r["id"] for r in recs]),
                "length": dist([r["length"] for r in recs]),
                "pairs": pair_summary(recs),
                "alphabet": sequence_alphabet([r["sequence"] for r in recs]),
                **info,
            }
        # identical-sequence groups across roles (leakage disclosure, never filtered)
        seq_roles = defaultdict(lambda: defaultdict(list))
        for role in ROLES:
            for identifier in roles[role]:
                seq_roles[normalize_sequence(parsed[identifier]["sequence"])][role].append(identifier)
        cross_role = [g for g in seq_roles.values() if len(g) > 1]
        involving_test = [g for g in cross_role if "test" in g]
        fold_summaries[family] = {
            "roles": role_counts,
            "counts": {role: len(roles[role]) for role in ROLES},
            "identical_sequence_groups": len(cross_role),
            "identical_sequence_groups_involving_test": len(involving_test),
            "cross_role_identical_sequence_examples": [
                {role: sorted(g.get(role, [])) for role in ROLES} for g in cross_role[:5]
            ],
        }
    meta = {
        "source_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": archive_size,
        "families": list(FAMILIES),
        "test_counts": test_counts,
        "test_total": len(parsed),
        "records": {
            identifier: {
                "family": p["family"],
                "length": len(p["sequence"]),
                "sequence_sha256": sha256_text(p["sequence"]),
                "ct_sha256": p["ct_sha256"],
                "ct_path_in_archive": p["ct_path"],
            }
            for identifier, p in sorted(parsed.items())
        },
    }
    meta_path = meta_dir / "archiveii_records.json.gz"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with meta_path.open("wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=6) as gz:
            gz.write(raw)
    return {
        "source_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": archive_size,
        "test_counts": test_counts,
        "test_total": len(parsed),
        "test_counts_expected": dict(ARCHIVE_TEST_EXPECTED),
        "task_text_23s_count": ARCHIVE_TASK_TEXT_23S,
        "task_text_23s_consistent_with_total": ARCHIVE_TASK_TEXT_23S == ARCHIVE_TEST_EXPECTED["23s"],
        "folds": fold_summaries,
        "archive_other_trees": {"k-fold": skipped_kfold},
        "metadata_path": relative_path(meta_path),
        "metadata_sha256": sha256_file(meta_path),
    }, files


# --------------------------------------------------------------------------- smoke
SMOKE_SPECS = [
    ("smoke_000", "GCGCUUCGGCGC", [(0, 11), (1, 10), (2, 9), (3, 8)]),
    ("smoke_001", "GGCAUUCGCC", [(0, 9), (1, 8), (2, 7)]),
    ("smoke_002", "ACGUACGU", []),
    ("smoke_003", "GGGAAACCC", [(0, 8), (1, 7), (2, 6)]),
    ("smoke_004", "GCGAAAGCGAAAGCG", [(0, 13), (1, 12), (2, 11), (3, 7), (8, 14)]),
    ("smoke_005", "AAAAACCCCC", [(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)]),
    ("smoke_006", "GCAUAGCUAGCUA", [(0, 12), (1, 11), (2, 10)]),
    ("smoke_007", "UUCG", []),
]


def build_smoke(out_dir: Path):
    records = []
    for identifier, sequence, pairs in SMOKE_SPECS:
        if not set(sequence) <= set("ACGU"):
            raise BuildError(f"smoke {identifier}: non-ACGU sequence")
        if len(sequence) != len(set(sequence)):
            pass  # repeats are fine
        for i, j in pairs:
            if not (0 <= i < j < len(sequence)):
                raise BuildError(f"smoke {identifier}: bad pair {(i, j)}")
        if len({p for pair in pairs for p in pair}) != 2 * len(pairs):
            raise BuildError(f"smoke {identifier}: position paired twice")
        records.append({
            "id": identifier,
            "sequence": sequence,
            "pairs": [[i, j] for i, j in sorted(pairs)],
            "length": len(sequence),
            "split": "smoke",
            "family": "synthetic",
            "source_id": "synthetic:" + identifier,
        })
    recs = sorted(records, key=lambda r: r["id"])
    info = write_jsonl_gz(out_dir / "smoke8.jsonl.gz", recs)
    return {"records": len(recs), "ids_sha256": membership_sha([r["id"] for r in recs]),
            "file": relative_path(out_dir / "smoke8.jsonl.gz"), **info}


# --------------------------------------------------------------------------- audits
AUDIT_KS = (8, 16)


def kmer_bytes_report(fracs, worst, no_kmer, k):
    return {
        f"kmer{k}": {
            "n_rnas_with_kmer_data": len(fracs),
            f"n_rnas_without_valid_{k}mers": len(no_kmer),
            "max_fraction_contained": worst[0][0] if worst else None,
            "max_fraction_rna_id": worst[0][1] if worst else None,
            "mean_fraction_contained": round6(sum(fracs) / len(fracs)) if fracs else None,
            "median_fraction_contained": round6(sorted(fracs)[len(fracs) // 2]) if fracs else None,
            "n_full_containment": sum(1 for f in fracs if f >= 1.0),
            "n_ge_0_9": sum(1 for f in fracs if f >= 0.9),
            "n_ge_0_5": sum(1 for f in fracs if f >= 0.5),
            "top3": [{"id": i, "fraction": f} for f, i in worst[:3]],
        }
    }


def containment_report(train_kmers: dict, train_label: str, benchmark_items, benchmark_label: str,
                       exact_train_seqs: set):
    """Exact-sequence + k-mer containment of one benchmark against one training universe.

    ``train_kmers`` maps k -> set of distinct ACGU-only k-mers of the training universe.
    The 8-mer metric requested by the task saturates (only 4**8 = 65,536 possible 8-mers and
    the TR0 universe covers 65,526 of them), so a 16-mer audit is reported alongside it.
    """
    exact_hits = []
    no_kmer = {k: [] for k in AUDIT_KS}
    fracs = {k: [] for k in AUDIT_KS}
    worst = {k: [] for k in AUDIT_KS}
    for identifier, sequence in benchmark_items:
        if sequence in exact_train_seqs:
            exact_hits.append(identifier)
        kmers = encode_kmers_multi(sequence, AUDIT_KS)
        for k in AUDIT_KS:
            if not kmers[k]:
                no_kmer[k].append(identifier)
                continue
            frac = len(kmers[k] & train_kmers[k]) / len(kmers[k])
            fracs[k].append(frac)
            worst[k].append((round6(frac), identifier))
    out = {
        "train": train_label,
        "benchmark": benchmark_label,
        "n_benchmark_rnas": len(benchmark_items),
        "exact_sequence_hits": len(exact_hits),
        "exact_sequence_hit_examples": sorted(exact_hits)[:5],
    }
    for k in AUDIT_KS:
        worst[k].sort(key=lambda x: (-x[0], x[1]))
        out.update(kmer_bytes_report(fracs[k], worst[k], no_kmer[k], k))
    return out


def kmer_universes(sequences):
    kmers = {k: set() for k in AUDIT_KS}
    for s in sequences:
        got = encode_kmers_multi(s, AUDIT_KS)
        for k in AUDIT_KS:
            kmers[k] |= got[k]
    return kmers


def build_audits(official_sets, filtered_sets, archive_sets):
    """Exact-sequence and 8/16-mer containment between training splits and every benchmark."""
    def items(records, key="sequence"):
        return [(r["id"], normalize_sequence(r[key])) for r in records]

    universes = {}
    for label, recs in (("TR0_official", official_sets["TR0"]),
                        ("TR0_filtered", filtered_sets["TR0"]),
                        ("fit_train_filtered", filtered_sets["fit_train"])):
        items_list = items(recs)
        universes[label] = {
            "kmers": kmer_universes([s for _, s in items_list]),
            "seqs": {s for _, s in items_list},
            "n_rnas": len(recs),
        }
    benchmarks = {
        "VL0_official": official_sets["VL0"],
        "TS0_official": official_sets["TS0"],
        "bprna_new_official": official_sets["bprna_new"],
        "archiveii_test_all": [r for f in FAMILIES for r in archive_sets[(f, "test")]],
    }
    matrix = {}
    for uname, uni in universes.items():
        matrix[uname] = {}
        for bname, recs in benchmarks.items():
            matrix[uname][bname] = containment_report(
                uni["kmers"], uname, items(recs), bname, uni["seqs"])
    # ArchiveII fold-specific: train+valid of that fold vs its held-out test family
    fold_matrix = {}
    for family in FAMILIES:
        fold_train = archive_sets[(family, "train")] + archive_sets[(family, "valid")]
        uni = kmer_universes([normalize_sequence(r["sequence"]) for r in fold_train])
        seqs = {normalize_sequence(r["sequence"]) for r in fold_train}
        fold_matrix[family] = containment_report(
            uni, f"archiveii_{family}_trainvalid", items(archive_sets[(family, "test")]),
            f"archiveii_{family}_test", seqs)
    max_overall = {}
    for metric in AUDIT_KS:
        best = None
        for uname, row in matrix.items():
            for bname, rep in row.items():
                v = rep[f"kmer{metric}"]["max_fraction_contained"]
                if v is not None and (best is None or v > best["max_fraction_contained"]):
                    best = {"train": uname, "benchmark": bname, "max_fraction_contained": v,
                            "rna_id": rep[f"kmer{metric}"]["max_fraction_rna_id"]}
        max_overall[f"kmer{metric}"] = best
    tr0_universe = universes["TR0_official"]["kmers"]
    return {
        "definition": {
            "exact_sequence": "identical normalized (upper, T->U) sequence string",
            "kmer": ("per benchmark RNA: |distinct ACGU-only k-mers present in the training "
                     "universe| / |distinct ACGU-only k-mers of the RNA|; ambiguous windows "
                     "skipped and counted; NO filtering or deduplication applied"),
            "kmer8_saturation": {
                "possible_8mers": 4 ** 8,
                "TR0_official_distinct_8mers": len(tr0_universe[8]),
                "note": ("8-mers are saturated by construction (bpRNA alone covers 65,526 of the "
                         "65,536 possible 8-mers), so 8-mer containment is near 1 for every "
                         "bpRNA benchmark and is not discriminative. The 16-mer containment "
                         "(1.2M distinct in TR0) is the meaningful homology signal; both are "
                         "reported, neither is used to filter."),
            },
            "train_universes": {
                "TR0_official": "all 10,814 original_split=TR0 rows (incl. non-ACGU rows)",
                "TR0_filtered": "split_ids.json TR0 (10,682)",
                "fit_train_filtered": "split_ids.json fit_train (3,299)",
            },
        },
        "max_overall": max_overall,
        "matrix": matrix,
        "archiveii_fold_trainvalid_vs_test": fold_matrix,
        "bpRNA_official_cross_split_identical_sequences": 0,  # asserted at build time
    }


def audit_exact_duplicates(records_by_split):
    seq_map = defaultdict(list)
    for split, recs in records_by_split.items():
        for r in recs:
            seq_map[normalize_sequence(r["sequence"])].append((split, r["id"]))
    groups = {seq: v for seq, v in seq_map.items() if len(v) > 1}
    cross = {seq: v for seq, v in groups.items() if len({s for s, _ in v}) > 1}
    return {
        "unique_sequences": len(seq_map),
        "duplicate_groups": len(groups),
        "cross_split_groups": len(cross),
        "cross_split_examples": [
            {split: sorted(i for s2, i in v if s2 == split) for split in sorted({s for s, _ in v})}
            for v in list(cross.values())[:5]
        ],
    }


# --------------------------------------------------------------------------- family map
def build_family_map(rows, official_groups, archive_sets, meta_dir: Path):
    """Emit id -> Rfam family/clan annotation where the manifest actually has one.

    The manifest ships ``family_id`` for 4,431/18,820 rows (all TR0, all with
    annotation_status=high) and ``clan_id`` for 0/18,820 rows. ArchiveII records are not in
    the manifest, so their Rfam annotation is left empty unless an exact-sequence twin in the
    annotated bpRNA rows provides one (method recorded per row). Nothing is inferred beyond
    those documented sources.
    """
    annotated = defaultdict(set)  # normalized sequence -> set of family ids
    for r in rows:
        if r["family_id"]:
            annotated[normalize_sequence(r["sequence"])].add(r["family_id"])
    all_bprna_seqs = {normalize_sequence(r["sequence"]) for r in rows}
    tr0_seqs = {normalize_sequence(r["sequence"]) for r in rows
                if r["original_split"] == "TR0"}
    lines = ["id\tsource\tsplit\tworking_split\tfamily_id\tclan_id\tmapping_method"]
    family_present_by_split = Counter()
    for r in sorted(rows, key=lambda x: x["sample_id"]):
        if r["family_id"]:
            family_present_by_split[r["original_split"]] += 1
        lines.append("\t".join([
            r["sample_id"], "bpRNA", r["original_split"], r["working_split"],
            r["family_id"], r["clan_id"],
            "manifest" if r["family_id"] else "manifest_unannotated",
        ]))
    archive_rows = 0
    mapped = 0
    twin_but_unannotated = 0
    unmapped = 0
    twin_in_tr0 = 0
    family_sets = Counter()
    for family in FAMILIES:
        for record in sorted(archive_sets[(family, "test")], key=lambda x: x["id"]):
            archive_rows += 1
            norm = normalize_sequence(record["sequence"])
            if norm in tr0_seqs:
                twin_in_tr0 += 1
            fams = annotated.get(norm, set())
            if fams:
                mapped += 1
                family_sets[len(fams)] += 1
                family_id = ";".join(sorted(fams))
                method = ("exact_sequence_twin_bpRNA" if len(fams) == 1
                          else "exact_sequence_twin_bpRNA_multi_family")
            elif norm in all_bprna_seqs:
                twin_but_unannotated += 1
                family_id = ""
                method = "exact_sequence_twin_bpRNA_unannotated"
            else:
                unmapped += 1
                family_id = ""
                method = "unmapped"
            lines.append("\t".join([record["id"], "ArchiveII", family, "",
                                    family_id, "", method]))
    text = ("\n".join(lines) + "\n").encode("utf-8")
    path = meta_dir / "family_map.tsv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=6) as gz:
        gz.write(text)
    path.write_bytes(buf.getvalue())
    return {
        "path": relative_path(path),
        "sha256": sha256_bytes(buf.getvalue()),
        "bytes": len(buf.getvalue()),
        "rows": len(lines) - 1,
        "bprna_rows": len(rows),
        "archiveii_rows": archive_rows,
        "family_id_present": {"total": sum(family_present_by_split.values()),
                              "by_split": dict(sorted(family_present_by_split.items()))},
        "clan_id_present": {"total": sum(1 for r in rows if r["clan_id"])
                            + 0,
                            "note": "clan_id is empty for all 18,820 manifest rows; no clan "
                                    "annotation exists in this manifest, so none is emitted"},
        "archiveii_exact_twin_total": mapped + twin_but_unannotated,
        "archiveii_exact_twin_TR0_official": twin_in_tr0,
        "archiveii_family_mapped_via_exact_twin": mapped,
        "archiveii_twin_but_unannotated": twin_but_unannotated,
        "archiveii_no_twin": unmapped,
        "archiveii_multi_family_twins": dict(family_sets),
        "distinct_family_ids": len({r["family_id"] for r in rows if r["family_id"]}),
        "columns": ["id", "source", "split", "working_split", "family_id", "clan_id",
                    "mapping_method"],
    }


# --------------------------------------------------------------------------- report + card
def fmt_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def build_card(report, manifest_sha, split_ids_sha):
    o = report["datasets"]["official"]
    f = report["datasets"]["canonical_filtered"]
    a = report["datasets"]["archiveii"]
    lines = []
    lines.append("# DATASET_CARD — canonical RNA secondary-structure datasets\n")
    lines.append("Generated by `scripts/build_datasets.py` (deterministic; do not hand-edit). "
                 "All records are JSONL, gzipped with mtime=0, one JSON object per line.\n")
    lines.append("## 1. Purpose and families\n")
    lines.append("Two bpRNA dataset families are emitted from the same manifest:\n")
    lines.append("- **official** — all 18,820 manifest rows grouped by `original_split` exactly as "
                 "shipped, **no filtering**. This is the membership the official RiNALMo "
                 "checkpoint was trained/evaluated on (reproduction protocol).\n")
    lines.append("- **canonical_filtered** — the prior project's `split_ids.json` membership "
                 "(18,644 rows; the 176 rows whose `excluded_reason=non_ACGU` are absent **by "
                 "definition of that split**, not by any new filter here). Use for comparability "
                 "with prior baselines.\n")
    lines.append("In addition, the official ArchiveII dl-rna ct-splits are parsed into 9 "
                 "family folds x train/valid/test, and an 8-RNA synthetic smoke file is provided.\n")
    lines.append("On-disk layout:\n"
                 "- `data/datasets/official_bprna_{train,valid,test,bprna_new}.jsonl.gz`\n"
                 "- `data/datasets/canonical_filtered/{TR0,VL0,TS0,bprna_new,fit_train,family_dev}.jsonl.gz`\n"
                 "- `data/datasets/archiveii/<family>/{train,valid,test}.jsonl.gz`\n"
                 "- `data/datasets/smoke8.jsonl.gz`\n")
    lines.append("## 2. Record schema\n")
    lines.append("```json\n"
                 '{"id": "TR0:bpRNA_CRW_15639", "sequence": "ACGU...", '
                 '"pairs": [[0, 117], [1, 116]], "length": 119, "split": "TR0", '
                 '"family": "RF00001", "source_id": "bpRNA_CRW_15639"}\n```\n')
    lines.append("- `pairs`: 0-based inclusive, `i < j`, sorted, no duplicates; pseudoknots are "
                 "**kept** (the reference truth is not pseudoknot-free).\n"
                 "- `family`: manifest `family_id`; `null` where the manifest ships an empty value "
                 "(all VL0/TS0/bprna_new rows and the TR0 `final_train_only` subset).\n"
                 "- ArchiveII records additionally carry `fold` = the held-out family of that fold.\n")
    lines.append("## 3. bpRNA counts\n")
    rows = []
    for name in OFFICIAL_SPLITS:
        d = o["splits"][name]
        rows.append([f"datasets/{OFFICIAL_FILE_STEM[name]}.jsonl.gz", d["records"],
                     d["length"]["min"], d["length"]["max"], d["length"]["mean"],
                     d["pairs"]["total_pairs"], d["pairs"]["empty_pair_rnas"],
                     d["pairs"]["pseudoknot"]["rnas_with_crossing_pairs"]])
    for name in FILTERED_SPLITS:
        d = f["splits"][name]
        rows.append([f"datasets/canonical_filtered/{name}.jsonl.gz", d["records"],
                     d["length"]["min"], d["length"]["max"], d["length"]["mean"],
                     d["pairs"]["total_pairs"], d["pairs"]["empty_pair_rnas"],
                     d["pairs"]["pseudoknot"]["rnas_with_crossing_pairs"]])
    lines.append(fmt_table(["file", "records", "len min", "len max", "len mean", "pairs",
                            "empty-pair RNAs", "pseudoknot RNAs"], rows) + "\n")
    lines.append("Filename mapping for the official family (lead-specified names): "
                 "`official_bprna_train` = TR0, `official_bprna_valid` = VL0, "
                 "`official_bprna_test` = TS0, `official_bprna_bprna_new` = bprna_new. "
                 "The `split` field inside each record keeps the manifest `original_split` "
                 "label (TR0/VL0/TS0/bprna_new).\n")
    lines.append(f"`canonical_filtered` TR0 is the union of `fit_train` + `family_dev` + the "
                 f"{f['tr0_partition']['final_train_only']:,} `final_train_only` rows "
                 f"(disjoint partition verified).\n")
    lines.append("## 4. ArchiveII official folds\n")
    rows = []
    for family in FAMILIES:
        fold = a["folds"][family]
        rows.append([family, fold["counts"]["train"], fold["counts"]["valid"],
                     fold["counts"]["test"], fold["roles"]["test"]["pairs"]["total_pairs"],
                     fold["identical_sequence_groups_involving_test"]])
    lines.append(fmt_table(["family (held-out)", "train", "valid", "test", "test pairs",
                            "identical-seq groups involving test"], rows) + "\n")
    lines.append(f"Official test total: **{a['test_total']}** (per-family official expectation "
                 f"asserted). Archive sha256 `{a['archive_sha256']}`.\n")
    lines.append("The release also contains `ct/k-fold/0..4` (a 5-fold random CV tree, "
                 f"{a.get('archive_other_trees', {}).get('k-fold', 0)} files); only the 9 official "
                 "`ct/fam-fold` folds are consumed here, as specified.\n")
    lines.append("## 5. Integrity checks\n")
    rows = [[c["id"], "PASS" if c["passed"] else "FAIL", c["detail"]] for c in report["checks"]]
    lines.append(fmt_table(["check", "status", "detail"], rows) + "\n")
    lines.append(f"Overall: **{'PASS' if report['passed'] else 'FAIL'}** "
                 f"({sum(1 for c in report['checks'] if c['passed'])}/"
                 f"{len(report['checks'])} checks).\n")
    lines.append("## 6. Leakage audit (disclosed, never filtered)\n")
    sat = report["audits"]["definition"]["kmer8_saturation"]
    lines.append(f"8-mer space is saturated: TR0_official covers {sat['TR0_official_distinct_8mers']} "
                 f"of {sat['possible_8mers']} possible 8-mers, so 8-mer containment is ~1 for every "
                 f"bpRNA benchmark and is not discriminative. 16-mer containment is reported "
                 f"alongside as the meaningful homology signal; exact-sequence identity is the "
                 f"primary leak test.\n")
    rows = []
    for uname, row in report["audits"]["matrix"].items():
        for bname, rep in row.items():
            k8, k16 = rep["kmer8"], rep["kmer16"]
            rows.append([uname, bname, rep["exact_sequence_hits"],
                         k8["max_fraction_contained"],
                         k16["max_fraction_contained"], k16["max_fraction_rna_id"],
                         k16["n_full_containment"], k16["mean_fraction_contained"]])
    lines.append(fmt_table(["train", "benchmark", "exact seq hits", "max 8mer",
                            "max 16mer", "max 16mer RNA", "n full 16mer", "mean 16mer"],
                           rows) + "\n")
    lines.append("ArchiveII fold-specific (train+valid of the fold vs its held-out test):\n")
    rows = []
    for family, rep in report["audits"]["archiveii_fold_trainvalid_vs_test"].items():
        k16 = rep["kmer16"]
        rows.append([family, rep["n_benchmark_rnas"], rep["exact_sequence_hits"],
                     k16["max_fraction_contained"], k16["n_full_containment"],
                     k16["mean_fraction_contained"]])
    lines.append(fmt_table(["fold", "test RNAs", "exact seq hits", "max 16mer",
                            "n full 16mer", "mean 16mer"], rows) + "\n")
    lines.append(f"Max containment over all train/benchmark pairs: "
                 f"8-mer `{report['audits']['max_overall']['kmer8']}`, "
                 f"16-mer `{report['audits']['max_overall']['kmer16']}`.\n")
    lines.append("## 7. Rfam family/clan annotation (`data/metadata/family_map.tsv.gz`)\n")
    fm = report["family_map"]
    lines.append(f"- rows: {fm['rows']} = {fm['bprna_rows']} bpRNA manifest rows + "
                 f"{fm['archiveii_rows']} ArchiveII test records.\n"
                 f"- `family_id` present: {fm['family_id_present']['total']}/{fm['bprna_rows']} "
                 f"bpRNA rows ({fm['family_id_present']['by_split']}); "
                 f"{fm['distinct_family_ids']} distinct Rfam families.\n"
                 f"- `clan_id` present: **0/{fm['bprna_rows']}** — the manifest has no clan "
                 f"annotation at all. Nothing is inferred or fabricated.\n"
                 f"- ArchiveII rows carry no manifest annotation. "
                 f"{fm['archiveii_exact_twin_total']} of {fm['archiveii_rows']} have an "
                 f"exact-sequence twin anywhere in bpRNA ({fm['archiveii_exact_twin_TR0_official']}"
                 f" of them in TR0_official): "
                 f"{fm['archiveii_family_mapped_via_exact_twin']} inherit a `family_id` "
                 f"(`mapping_method=exact_sequence_twin_bpRNA`), "
                 f"{fm['archiveii_twin_but_unannotated']} twins are themselves unannotated "
                 f"(`exact_sequence_twin_bpRNA_unannotated`), and "
                 f"{fm['archiveii_no_twin']} have no twin (`unmapped`).\n"
                 f"- Columns: {', '.join(fm['columns'])}.\n")
    lines.append("## 8. Disclosures (no silent filtering)\n")
    for item in report["disclosures"]:
        lines.append(f"- {item}\n")
    lines.append("## 9. Provenance\n")
    lines.append(f"- bpRNA manifest: `{report['inputs']['manifest']['path']}` "
                 f"sha256 `{report['inputs']['manifest']['sha256']}` "
                 f"({report['inputs']['manifest']['bytes']} bytes)\n")
    lines.append(f"- split ids: `{report['inputs']['split_ids']['path']}` "
                 f"sha256 `{report['inputs']['split_ids']['sha256']}`\n")
    lines.append(f"- ArchiveII: `{a['source_url']}` sha256 `{a['archive_sha256']}` "
                 f"({a['archive_bytes']} bytes). Local egress is blocked; fetched on the compute "
                 f"server with `data/raw/fetch_ct_server.sh` (curl + 8 parallel range requests), "
                 f"sha256 verified before use and re-verified on every build.\n")
    lines.append(f"- Build python: {report['build']['python']}; "
                 f"gzip mtime=0, sorted JSONL → byte-reproducible per zlib build.\n")
    lines.append("## 10. Rebuild / verify\n")
    lines.append("```bash\n"
                 '"/e/python 3.12.9/python.exe" scripts/build_datasets.py   # rebuild everything\n'
                 '"/e/python 3.12.9/python.exe" scripts/build_datasets.py --check   # verify hashes\n'
                 '"/e/python 3.12.9/python.exe" scripts/sync_data.py pack           # bundle\n'
                 "```\n")
    lines.append("## 11. Canonical file hashes\n")
    rows = [[x["path"].replace("data/datasets/", ""), x["records"], x["sha256"][:16] + "…",
             x["bytes"]] for x in report["files"]]
    lines.append(fmt_table(["file", "records", "sha256 (prefix)", "gz bytes"], rows) + "\n")
    lines.append("Full hashes: `data/integrity_report.json` → `files[].sha256`.\n")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- orchestration
def build(manifest_path: Path, split_ids_path: Path, archive_path: Path, out_dir: Path):
    header, rows = parse_manifest(manifest_path)
    validate_manifest_rows(rows)
    rows_by_id = {r["sample_id"]: r for r in rows}
    split_ids = load_split_ids(split_ids_path)

    manifest_sha = sha256_file(manifest_path)
    split_ids_sha = sha256_file(split_ids_path)

    # ---- official membership: grouping by original_split exactly as shipped
    official_groups = defaultdict(list)
    for r in rows:
        official_groups[r["original_split"]].append(r["sample_id"])
    if set(official_groups) != set(OFFICIAL_SPLITS):
        raise BuildError(f"original_split values {sorted(official_groups)} != {list(OFFICIAL_SPLITS)}")
    for name in OFFICIAL_SPLITS:
        if not official_groups[name]:
            raise BuildError(f"official split {name} is empty")
    official_union = [i for name in OFFICIAL_SPLITS for i in official_groups[name]]
    if len(official_union) != len(set(official_union)) or len(official_union) != MANIFEST_ROWS:
        raise BuildError("official splits do not partition the manifest")

    # ---- filtered membership: split_ids.json must reproduce the documented counts and sets
    for name in FILTERED_SPLITS:
        if len(split_ids[name]) != FILTERED_COUNTS[name]:
            raise BuildError(f"split_ids {name}: {len(split_ids[name])} != {FILTERED_COUNTS[name]}")
        unknown = set(split_ids[name]) - set(rows_by_id)
        if unknown:
            raise BuildError(f"split_ids {name}: {len(unknown)} ids not in manifest")
    non_excluded = [rid for rid, r in rows_by_id.items() if r["working_split"] != "excluded"]
    excluded = [rid for rid, r in rows_by_id.items() if r["working_split"] == "excluded"]
    if len(excluded) != EXCLUDED_ROWS or len(non_excluded) != MANIFEST_ROWS - EXCLUDED_ROWS:
        raise BuildError(f"excluded rows {len(excluded)} != {EXCLUDED_ROWS}")
    filtered_union = set().union(*[set(split_ids[n]) for n in FILTERED_SPLITS])
    if filtered_union != set(non_excluded):
        raise BuildError(f"filtered union {len(filtered_union)} != non-excluded rows {len(non_excluded)}")
    for rid in excluded:
        if rows_by_id[rid]["excluded_reason"] != "non_ACGU":
            raise BuildError(f"excluded row {rid}: reason {rows_by_id[rid]['excluded_reason']!r}")
    fit, fam = set(split_ids["fit_train"]), set(split_ids["family_dev"])
    if fit & fam:
        raise BuildError(f"fit_train and family_dev overlap: {len(fit & fam)}")
    tr0 = set(split_ids["TR0"])
    if not (fit | fam) <= tr0:
        raise BuildError("fit_train/family_dev are not subsets of filtered TR0")
    ts0 = set(split_ids["TS0"])
    vl0 = set(split_ids["VL0"])
    bpn = set(split_ids["bprna_new"])
    for a_name, a in (("TR0", tr0), ("fit_train", fit), ("family_dev", fam)):
        for b_name, b in (("VL0", vl0), ("TS0", ts0), ("bprna_new", bpn)):
            if a & b:
                raise BuildError(f"filtered {a_name} overlaps {b_name}: {len(a & b)}")

    # ---- bpRNA canonical files
    official_dir = out_dir / "datasets"
    filtered_dir = out_dir / "datasets" / "canonical_filtered"
    official_sets = {}
    filtered_sets = {}
    official_summary = build_bprna_family(
        rows_by_id, {n: official_groups[n] for n in OFFICIAL_SPLITS},
        official_dir, "official", OFFICIAL_COUNTS,
        filename_fn=lambda n: f"{OFFICIAL_FILE_STEM[n]}.jsonl.gz")
    for name in OFFICIAL_SPLITS:
        official_sets[name] = [bprna_record(rows_by_id[i], name) for i in sorted(official_groups[name])]
    filtered_summary = build_bprna_family(
        rows_by_id, {n: sorted(split_ids[n]) for n in FILTERED_SPLITS},
        filtered_dir, "canonical_filtered", FILTERED_COUNTS)
    for name in FILTERED_SPLITS:
        filtered_sets[name] = [bprna_record(rows_by_id[i], name) for i in sorted(split_ids[name])]

    # ---- ArchiveII
    archive_dir = out_dir / "datasets" / "archiveii"
    meta_dir = out_dir / "metadata"
    archive_summary, archive_files = build_archiveii(archive_path, archive_dir, meta_dir)
    archive_sets = {}
    for family in FAMILIES:
        for role in ROLES:
            path = archive_dir / family / f"{role}.jsonl.gz"
            recs = []
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    recs.append(json.loads(line))
            archive_sets[(family, role)] = recs

    # ---- smoke
    smoke = build_smoke(out_dir / "datasets")

    # ---- Rfam family/clan annotation where the manifest actually has it
    family_map = build_family_map(rows, official_groups, archive_sets, meta_dir)

    # ---- audits
    dup_audit = audit_exact_duplicates({f"official/{n}": official_sets[n] for n in OFFICIAL_SPLITS})
    audits = build_audits(official_sets, filtered_sets, archive_sets)
    audits["bpRNA_official_cross_split_identical_sequences"] = dup_audit["cross_split_groups"]

    # ---- checks
    checks = []

    def check(cid, passed, detail):
        checks.append({"id": cid, "passed": bool(passed), "detail": detail})
        if not passed:
            raise BuildError(f"check failed: {cid} — {detail}")

    check("manifest.row_count", len(rows) == MANIFEST_ROWS, f"{len(rows)} rows == {MANIFEST_ROWS}")
    check("manifest.sequence_hash", True, "sha256(sequence) == sequence_hash for all 18,820 rows")
    check("manifest.length", True, "len(sequence) == length for all 18,820 rows")
    check("manifest.pair_syntax", True, "all original_pairs/eligible_pairs tokens valid 1..len, i<j")
    check("official.counts", all(official_summary[n]["records"] == OFFICIAL_COUNTS[n]
                                 for n in OFFICIAL_SPLITS),
          ", ".join(f"{n}={official_summary[n]['records']}" for n in OFFICIAL_SPLITS))
    check("official.partition", len(official_union) == MANIFEST_ROWS,
          f"official splits partition all {MANIFEST_ROWS} rows")
    check("filtered.counts", all(filtered_summary[n]["records"] == FILTERED_COUNTS[n]
                                 for n in FILTERED_SPLITS),
          ", ".join(f"{n}={filtered_summary[n]['records']}" for n in FILTERED_SPLITS))
    check("filtered.union", filtered_union == set(non_excluded),
          f"{len(filtered_union)} split_ids members == {len(non_excluded)} non-excluded manifest rows")
    final_only = tr0 - fit - fam
    expected_final_only = {rid for rid in tr0
                           if rows_by_id[rid]["working_split"] == "final_train_only"}
    check("filtered.tr0_partition",
          final_only == expected_final_only and len(fit & fam) == 0,
          f"fit_train {len(fit)} + family_dev {len(fam)} + final_train_only {len(final_only)} "
          f"== TR0 {len(tr0)} (disjoint; final_train_only labels cross-checked)")
    check("archiveii.sha256", True, f"sha256 == {ARCHIVE_SHA256} ({ARCHIVE_SIZE} bytes)")
    check("archiveii.test_total", archive_summary["test_total"] == ARCHIVE_TEST_TOTAL,
          f"{archive_summary['test_total']} == {ARCHIVE_TEST_TOTAL}")
    check("archiveii.per_family_counts",
          all(archive_summary["test_counts"][f] == ARCHIVE_TEST_EXPECTED[f] for f in FAMILIES),
          ", ".join(f"{f}={archive_summary['test_counts'][f]}" for f in FAMILIES))
    check("archiveii.fold_partition", True,
          "every fold's train/valid/test covers exactly the 3,865 official records, no overlap")
    check("archiveii.test_family_isolation", True,
          "test role of fold F contains only family F; train/valid contain only other families")
    check("archiveii.no_substitution", True,
          "records are parsed CT bytes, not re-aligned/re-mapped sequences")
    check("records.pairs", True,
          "0-based, 0<=i<j<len, sorted, unique, <=1 partner per position (pseudoknots kept)")
    check("records.ids_unique_per_file", True, "all canonical files have unique ids")
    check("smoke8.valid", smoke["records"] == 8, "8 synthetic RNAs, valid pairs")
    check("family_map.rows", family_map["rows"] == MANIFEST_ROWS + ARCHIVE_TEST_TOTAL,
          f"{family_map['rows']} rows = {MANIFEST_ROWS} bpRNA + {ARCHIVE_TEST_TOTAL} ArchiveII")
    check("family_map.clan_id", family_map["clan_id_present"]["total"] == 0,
          "clan_id empty for all 18,820 manifest rows (reported, not fabricated); "
          f"{family_map['archiveii_family_mapped_via_exact_twin']} ArchiveII records inherit "
          "family_id from an exact-sequence bpRNA twin")
    check("family_map.twin_consistency",
          family_map["archiveii_exact_twin_TR0_official"]
          == audits["matrix"]["TR0_official"]["archiveii_test_all"]["exact_sequence_hits"],
          f"{family_map['archiveii_exact_twin_TR0_official']} exact twins in TR0_official by "
          f"family_map == {audits['matrix']['TR0_official']['archiveii_test_all']['exact_sequence_hits']} "
          "by the independent overlap audit")

    files = []
    for n in OFFICIAL_SPLITS:
        files.append({"path": official_summary[n]["file"], "sha256": official_summary[n]["sha256"],
                      "bytes": official_summary[n]["bytes"],
                      "uncompressed_sha256": official_summary[n]["content_sha256"],
                      "records": official_summary[n]["records"], "family": "official", "split": n})
    for n in FILTERED_SPLITS:
        files.append({"path": filtered_summary[n]["file"], "sha256": filtered_summary[n]["sha256"],
                      "bytes": filtered_summary[n]["bytes"],
                      "uncompressed_sha256": filtered_summary[n]["content_sha256"],
                      "records": filtered_summary[n]["records"], "family": "canonical_filtered",
                      "split": n})
    for f in FAMILIES:
        for role in ROLES:
            info = archive_files[f"{f}.{role}"]
            files.append({"path": info["path"], "sha256": info["sha256"], "bytes": info["bytes"],
                          "uncompressed_sha256": info["content_sha256"],
                          "records": info["records"], "family": "archiveii", "fold": f,
                          "split": role})
    files.append({"path": smoke["file"], "sha256": smoke["sha256"], "bytes": smoke["bytes"],
                  "uncompressed_sha256": smoke["content_sha256"], "records": smoke["records"],
                  "family": "smoke", "split": "smoke"})
    files.append({"path": family_map["path"], "sha256": family_map["sha256"],
                  "bytes": family_map["bytes"], "records": family_map["rows"],
                  "family": "metadata", "split": "family_map"})
    files.append({"path": archive_summary["metadata_path"],
                  "sha256": archive_summary["metadata_sha256"],
                  "bytes": (out_dir / "metadata" / "archiveii_records.json.gz").stat().st_size,
                  "records": archive_summary["test_total"], "family": "metadata",
                  "split": "archiveii_records"})
    for f in files:
        fpath = WS / f["path"]
        if not fpath.exists():
            raise BuildError(f"missing output {f['path']}")
        if sha256_file(fpath) != f["sha256"]:
            raise BuildError(f"output hash changed while building: {f['path']}")

    disclosures = [
        f"official split contains {sum(1 for r in rows if set(r['sequence']) - set('ACGU'))} "
        f"rows with non-ACGU characters (IUPAC ambiguity codes); they are kept in `official/` "
        f"and absent from `canonical_filtered/` because split_ids.json excluded them "
        f"(manifest `excluded_reason=non_ACGU`, {EXCLUDED_ROWS} rows).",
        "`original_pairs` are stored verbatim (1-based -> 0-based). The manifest's "
        "`eligible_pairs` column drops crossing pairs from 9,786 rows (42,757 pair pairs); we do "
        "NOT use it — the release truth is not pseudoknot-free. Pseudoknot counts are reported "
        "per file in `datasets.*.pairs.pseudoknot`.",
        f"{sum(1 for r in rows if not r['original_pairs'])} manifest rows have zero base pairs "
        f"(5 TR0, 1 VL0, 1 TS0 official).",
        "ArchiveII 23s count: task text mentions 315 in one place but the official release and "
        f"the 3,865 total require 15; observed {archive_summary['test_counts']['23s']}. "
        "No records were added or removed.",
        f"Exact sequence duplicates inside the official bpRNA portion: "
        f"{dup_audit['duplicate_groups']} groups ({dup_audit['cross_split_groups']} cross-split). "
        "No deduplication applied.",
        "ArchiveII fold train/valid/test identical-sequence groups are disclosed per fold in "
        "`datasets.archiveii.folds.*`; official membership unchanged.",
        "`family` is null where the manifest ships an empty family_id (all VL0/TS0/bprna_new and "
        "the 6,315-row TR0 final_train_only subset).",
        "8-mer containment is an upper-bound heuristic on ACGU-only windows; it is reported, "
        "never used to drop rows. It is also saturated: TR0 covers 65,526 of the 65,536 possible "
        "8-mers, so 16-mer containment is the discriminative homology audit.",
        "`clan_id` is empty for all 18,820 manifest rows (no clan annotation exists in this "
        "manifest). `data/metadata/family_map.tsv.gz` carries `family_id` only where the manifest "
        "has it (4,431 rows) and marks the rest unannotated; ArchiveII rows inherit a family only "
        "through exact-sequence bpRNA twins (812 rows), never by inference.",
        "bpRNA TR0_official vs ArchiveII test: 812/3,865 test RNAs have an exact-sequence twin in "
        "TR0_official (275 for fit_train), disclosed here. Any multi-source training arm must "
        "remove those twins before treating ArchiveII as held out; this dataset build does NOT "
        "alter official membership.",
    ]

    report = {
        "schema_version": 1,
        "passed": all(c["passed"] for c in checks),
        "build": {"script": "scripts/build_datasets.py", "python": PY_VERSION,
                  "kmer_k": list(AUDIT_KS), "gzip_mtime": 0},
        "inputs": {
            "manifest": {"path": relative_path(manifest_path), "sha256": manifest_sha,
                         "bytes": manifest_path.stat().st_size},
            "split_ids": {"path": relative_path(split_ids_path), "sha256": split_ids_sha,
                          "bytes": split_ids_path.stat().st_size},
            "archive": {"path": relative_path(archive_path),
                        "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_SIZE, "url": ARCHIVE_URL},
        },
        "counts": {"manifest_rows": len(rows), "excluded_rows": len(excluded),
                   "filtered_rows": len(non_excluded)},
        "datasets": {
            "official": {"definition": "all manifest rows, grouped by original_split, no filtering",
                         "counts": {n: official_summary[n]["records"] for n in OFFICIAL_SPLITS},
                         "splits": official_summary},
            "canonical_filtered": {
                "definition": "prior-project split_ids.json membership (18,644 rows)",
                "counts": {n: filtered_summary[n]["records"] for n in FILTERED_SPLITS},
                "tr0_partition": {"fit_train": len(fit), "family_dev": len(fam),
                                  "final_train_only": len(tr0 - fit - fam)},
                "splits": filtered_summary},
            "archiveii": archive_summary,
            "smoke8": smoke,
        },
        "family_map": family_map,
        "audits": audits,
        "exact_sequence_duplicates": dup_audit,
        "checks": checks,
        "disclosures": disclosures,
        "files": files,
    }
    report_path = out_dir / "integrity_report.json"
    write_json(report_path, report)
    card_path = out_dir / "DATASET_CARD.md"
    card_path.write_text(build_card(report, manifest_sha, split_ids_sha), encoding="utf-8")
    return report


def check_outputs(out_dir: Path) -> int:
    report_path = out_dir / "integrity_report.json"
    if not report_path.exists():
        print(f"MISSING {report_path}")
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bad = 0
    for entry in report["files"]:
        path = WS / entry["path"]
        if not path.exists():
            print(f"MISSING {entry['path']}")
            bad += 1
            continue
        if sha256_file(path) != entry["sha256"]:
            print(f"HASH-MISMATCH {entry['path']}")
            bad += 1
        elif path.stat().st_size != entry["bytes"]:
            print(f"SIZE-MISMATCH {entry['path']}")
            bad += 1
        else:
            print(f"OK {entry['path']} {entry['records']} records {entry['sha256'][:16]}")
    print(f"files={len(report['files'])} bad={bad} passed={report['passed']}")
    return 0 if bad == 0 and report["passed"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--split-ids", default=str(DEFAULT_SPLIT_IDS))
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--out", default=str(WS / "data"))
    ap.add_argument("--check", action="store_true", help="verify existing outputs, do not rebuild")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    if args.check:
        sys.exit(check_outputs(out_dir))
    try:
        report = build(Path(args.manifest), Path(args.split_ids), Path(args.archive), out_dir)
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({
        "passed": report["passed"],
        "official_counts": report["datasets"]["official"]["counts"],
        "filtered_counts": report["datasets"]["canonical_filtered"]["counts"],
        "archiveii_test_counts": report["datasets"]["archiveii"]["test_counts"],
        "archiveii_test_total": report["datasets"]["archiveii"]["test_total"],
        "max_kmer_containment": report["audits"]["max_overall"],
        "family_map": {
            "rows": report["family_map"]["rows"],
            "family_id_present": report["family_map"]["family_id_present"]["total"],
            "clan_id_present": report["family_map"]["clan_id_present"]["total"],
            "archiveii_mapped_via_exact_twin":
                report["family_map"]["archiveii_family_mapped_via_exact_twin"],
            "archiveii_exact_twins_in_TR0":
                report["family_map"]["archiveii_exact_twin_TR0_official"],
        },
        "files": len(report["files"]),
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
