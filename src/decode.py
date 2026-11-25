"""src/decode.py -- deterministic greedy base-pair decoding (0-based pairs).

Semantics are the frozen protocol used by every historical prediction file and
by the official RiNALMo evaluator:

* legal candidates: canonical AU/UA/GC/CG/GU/UG with ``j - i >= min_sep`` (4);
* ``score = sigmoid(logit)``; only candidates **strictly greater** than the
  threshold survive (``value <= threshold`` breaks the ranking);
* accept in descending probability order, ties broken by ``i`` then ``j``
  ascending, so decoding is deterministic and reproducible;
* each nucleotide takes at most one partner; crossing pairs are allowed (the
  references are not pseudoknot-free -- 16.7% of family-dev RNAs cross).

The greedy ranking rule is *exactly* what the official
``rinalmo.utils.sec_struct.prob_mat_to_sec_struct`` does (threshold + descending
probability + row/column cleaning); ``tests/test_official.py`` checks the two
implementations agree when the upstream package is importable.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import math

import numpy as np

CANONICAL_PAIRS = frozenset(
    {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
)
MIN_SEP = 4
VALID_BASES = frozenset("ACGU")
Pair = Tuple[int, int]


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


def validate_sequence(sequence: str, strict: bool = False) -> str:
    """Normalise an RNA sequence for decoding.

    Ambiguous IUPAC codes (N, R, Y, ...) are allowed and simply never form a
    canonical pair (exactly what the official decoder does via its canonical
    mask); only an empty sequence is an error. ``strict=True`` keeps the old
    behaviour for callers that need an alphabet guarantee.
    """
    sequence = normalize_sequence(sequence)
    if not sequence:
        raise ValueError("sequence must be a non-empty string")
    if strict:
        invalid = set(sequence) - VALID_BASES
        if invalid:
            raise ValueError(f"sequence contains invalid bases: {sorted(invalid)}")
    return sequence


def is_legal_pair(sequence: str, i: int, j: int, min_sep: int = MIN_SEP) -> bool:
    sequence = normalize_sequence(sequence)
    return (
        0 <= i < j < len(sequence)
        and (j - i) >= min_sep
        and (sequence[i], sequence[j]) in CANONICAL_PAIRS
    )


def candidate_pairs(sequence: str, min_sep: int = MIN_SEP) -> List[Pair]:
    """All legal candidate pairs ``(i, j)`` with ``i < j``, ascending."""
    sequence = validate_sequence(sequence)
    length = len(sequence)
    pairs: List[Pair] = []
    for i in range(length):
        base_i = sequence[i]
        for j in range(i + min_sep, length):
            if (base_i, sequence[j]) in CANONICAL_PAIRS:
                pairs.append((i, j))
    return pairs


def candidate_mask(sequence: str, min_sep: int = MIN_SEP) -> np.ndarray:
    sequence = normalize_sequence(sequence)
    mask = np.zeros((len(sequence), len(sequence)), dtype=bool)
    for i, j in candidate_pairs(sequence, min_sep):
        mask[i, j] = mask[j, i] = True
    return mask


def _candidate_arrays(sequence: str, min_sep: int) -> Tuple[np.ndarray, np.ndarray]:
    pairs = candidate_pairs(sequence, min_sep)
    if not pairs:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    indices = np.asarray(pairs, dtype=np.int64)
    return indices[:, 0], indices[:, 1]


def _validate_scores(scores: np.ndarray, sequence: str) -> np.ndarray:
    scores = np.asarray(scores)
    length = len(sequence)
    if scores.shape != (length, length):
        raise ValueError(f"scores must have shape ({length}, {length}), got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    return scores


def probabilities(scores: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scores))


def _ranked_candidates(
    scores: np.ndarray, sequence: str, min_sep: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate arrays ranked by ``(-probability, i, j)``."""
    scores = _validate_scores(scores, sequence)
    probs = probabilities(scores)
    cand_i, cand_j = _candidate_arrays(sequence, min_sep)
    if cand_i.size == 0:
        return cand_i, cand_j, np.zeros(0, dtype=np.float64)
    values = probs[cand_i, cand_j]
    order = np.lexsort((cand_j, cand_i, -values))
    return cand_i[order], cand_j[order], values[order]


def decode(
    scores: np.ndarray,
    sequence: str,
    threshold: float,
    min_sep: int = MIN_SEP,
) -> List[Pair]:
    """Greedy decode one symmetric logit matrix into a sorted 0-based pair list."""
    cand_i, cand_j, values = _ranked_candidates(scores, sequence, min_sep)
    if cand_i.size == 0:
        return []
    used = np.zeros(len(sequence), dtype=bool)
    pairs: List[Pair] = []
    for i, j, value in zip(cand_i, cand_j, values):
        if value <= threshold:
            break
        if not used[i] and not used[j]:
            used[i] = used[j] = True
            pairs.append((int(i), int(j)))
    return sorted(pairs)


def decode_multi(
    scores: np.ndarray,
    sequence: str,
    thresholds: Iterable[float],
    min_sep: int = MIN_SEP,
) -> Dict[float, List[Pair]]:
    """Decode several thresholds with one candidate ranking (validation sweeps)."""
    threshold_list = [float(t) for t in thresholds]
    output: Dict[float, List[Pair]] = {threshold: [] for threshold in threshold_list}
    cand_i, cand_j, values = _ranked_candidates(scores, sequence, min_sep)
    if cand_i.size == 0:
        return output
    for threshold in threshold_list:
        used = np.zeros(len(sequence), dtype=bool)
        pairs: List[Pair] = []
        for i, j, value in zip(cand_i, cand_j, values):
            if value <= threshold:
                break
            if not used[i] and not used[j]:
                used[i] = used[j] = True
                pairs.append((int(i), int(j)))
        output[threshold] = sorted(pairs)
    return output


def decode_from_logits_tensor(
    logits,
    sequence: str,
    threshold: float,
    min_sep: int = MIN_SEP,
) -> List[Pair]:
    array = logits.detach().to("cpu").float().numpy()
    return decode(array, sequence, threshold, min_sep)


def count_candidates(sequence: str, min_sep: int = MIN_SEP) -> int:
    return len(candidate_pairs(sequence, min_sep))


# ---------------------------------------------------------------------------
# Threshold grids and selection (validation only -- never fit on test)
# ---------------------------------------------------------------------------
def threshold_grid(
    thresholds: Sequence[float] | str | None,
    *,
    start: float = 0.01,
    stop: float = 0.29,
    step: float = 0.01,
) -> List[float]:
    """Parse ``[0.01, 0.05]`` / ``"0.01:0.29:0.01"`` / ``None`` into a sorted grid.

    The default is exactly the official RiNALMo grid
    ``[i / 100 for i in range(1, 30)]`` = 0.01 .. 0.29, which is what the
    published fine-tuned numbers selected their threshold on.
    """

    def from_range(low: float, high: float, stride: float) -> List[float]:
        if stride <= 0:
            raise ValueError("threshold step must be positive")
        if high < low:
            raise ValueError(f"threshold stop {high} < start {low}")
        count = int(math.floor((high - low) / stride + 1e-9))
        return [round(low + index * stride, 6) for index in range(count + 1)]

    if thresholds is None:
        values = from_range(start, stop, step)
    elif isinstance(thresholds, str):
        parts = [part for part in thresholds.split(":") if part.strip()]
        if len(parts) == 1:
            values = [round(float(parts[0]), 6)]
        elif len(parts) == 2:
            values = from_range(float(parts[0]), float(parts[1]), step)
        elif len(parts) == 3:
            values = from_range(float(parts[0]), float(parts[1]), float(parts[2]))
        else:
            raise ValueError(f"bad threshold spec {thresholds!r}")
    else:
        values = [round(float(value), 6) for value in thresholds]
    grid = sorted(set(values))
    if not grid:
        raise ValueError("empty threshold grid")
    for value in grid:
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold {value} outside (0, 1)")
    return grid


def select_threshold(
    metrics_by_threshold: Dict[float, Mapping[str, float]],
    metric: str = "rna_macro_exact_f1",
) -> Tuple[float, float]:
    """Pick the best threshold; ties resolve to the **smaller** threshold."""
    best_threshold = None
    best_value = float("-inf")
    for threshold in sorted(metrics_by_threshold):
        value = float(metrics_by_threshold[threshold].get(metric, float("-inf")))
        if best_threshold is None or value > best_value:
            best_threshold, best_value = threshold, value
    if best_threshold is None:
        raise ValueError("no thresholds to select from")
    return best_threshold, best_value
