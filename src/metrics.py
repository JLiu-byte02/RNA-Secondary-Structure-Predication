"""src/metrics.py -- pair-level metrics: exact + flexible F1, RNA- and family-macro.

Three clearly separated definitions live here, because a paper table needs all
of them and mixing them up is how numbers become unreproducible:

``pair_prf`` / ``rna_macro`` / ``family_macro``
    Exact set-based P/R/F1 on 0-based pairs.  Empty conventions mirror the prior
    audited project: both empty -> P=R=F1=1; exactly one side empty -> 0.
    Family-macro excludes placeholder families (``None``, ``"unknown"``,
    ``"NA"``, ``""``), averages per RNA inside a family, then across families.

``flexible_prf``
    "Shift-1" flexible matching: a predicted pair ``(i, j)`` matches a reference
    pair ``(i', j')`` iff ``|i-i'| <= radius`` and ``|j-j'| <= radius`` (radius 1),
    with a **one-to-one** deterministic greedy assignment (closest pairs first,
    then lexicographic).  Each reference pair can absorb at most one predicted
    pair, so the score stays a proper F1.

``official_flexible_prf``
    The published RiNALMo selection metric (``ss_f1`` with
    ``allow_flexible_pairings=True``): precision counts predictions within one
    position of *any* reference pair, recall counts references within one
    position of *any* prediction (not one-to-one; neighbourhood-set count).  This
    is what the official recipe tunes its threshold on, so it is our default
    selection metric; exact F1 remains the primary reported number.

``official_*`` (in :mod:`src.official`)
    Verbatim re-implementation of ``rinalmo.utils.sec_struct`` ``ss_precision`` /
    ``ss_recall`` / ``ss_f1`` -- precision relaxes the *target*, recall relaxes
    the *prediction*.  Use it only for reproducing published numbers.

All functions take 0-based ``(i, j)`` pairs with ``i < j`` and never assume a
particular length.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

Pair = Tuple[int, int]
Triple = Tuple[float, float, float]

EXCLUDED_FAMILIES = (None, "unknown", "NA", "")


# ---------------------------------------------------------------------------
# Exact metrics
# ---------------------------------------------------------------------------
def normalize_pairs(pairs: Iterable[Sequence[int]]) -> set[Pair]:
    """Coerce to a set of ordered 0-based pairs (``i < j``)."""
    normalized: set[Pair] = set()
    for pair in pairs:
        i, j = int(pair[0]), int(pair[1])
        normalized.add((i, j) if i < j else (j, i))
    return normalized


def pair_prf(reference: Iterable[Pair], predicted: Iterable[Pair]) -> Triple:
    """Exact base-pair ``(precision, recall, f1)`` with frozen empty conventions."""
    reference = normalize_pairs(reference)
    predicted = normalize_pairs(predicted)
    if not reference and not predicted:
        return 1.0, 1.0, 1.0
    if not reference or not predicted:
        return 0.0, 0.0, 0.0
    true_positive = len(reference & predicted)
    precision = true_positive / len(predicted)
    recall = true_positive / len(reference)
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def rna_macro(per_rna: Sequence[Triple]) -> Triple:
    """Mean of per-RNA ``(precision, recall, f1)`` triples."""
    if not per_rna:
        return 0.0, 0.0, 0.0
    count = len(per_rna)
    return (
        sum(value[0] for value in per_rna) / count,
        sum(value[1] for value in per_rna) / count,
        sum(value[2] for value in per_rna) / count,
    )


def family_macro(
    per_rna_with_family: Iterable[Tuple[Hashable, float]],
    excluded: Iterable[object] = EXCLUDED_FAMILIES,
) -> float:
    """Mean across known families of the within-family mean F1."""
    excluded = tuple(excluded)
    buckets: Dict[Hashable, List[float]] = defaultdict(list)
    for family, f1 in per_rna_with_family:
        if family not in excluded:
            buckets[family].append(float(f1))
    if not buckets:
        return 0.0
    family_means = [sum(values) / len(values) for values in buckets.values()]
    return sum(family_means) / len(family_means)


def per_family_f1(
    rows: Iterable[Tuple[Hashable, Iterable[Pair], Iterable[Pair]]],
    excluded: Iterable[object] = EXCLUDED_FAMILIES,
) -> Dict[str, Dict[str, float]]:
    """``{family: {"count": n, "f1": mean_per_rna_f1}}`` from raw pair rows."""
    excluded = tuple(excluded)
    buckets: Dict[Hashable, List[float]] = defaultdict(list)
    for family, reference, predicted in rows:
        if family in excluded:
            continue
        buckets[family].append(pair_prf(reference, predicted)[2])
    return {
        str(family): {"count": len(values), "f1": sum(values) / len(values)}
        for family, values in buckets.items()
    }


def empty_convention_stats(per_rna_flags: Iterable[Tuple[bool, bool]]) -> Dict[str, int]:
    """Count the empty-reference / empty-prediction convention cases."""
    flags = list(per_rna_flags)
    return {
        "both_empty": sum(1 for reference_empty, pred_empty in flags if reference_empty and pred_empty),
        "only_reference_empty": sum(
            1 for reference_empty, pred_empty in flags if reference_empty and not pred_empty
        ),
        "only_prediction_empty": sum(
            1 for reference_empty, pred_empty in flags if pred_empty and not reference_empty
        ),
    }


# ---------------------------------------------------------------------------
# Flexible ("shift-1") metrics
# ---------------------------------------------------------------------------
def flexible_matches(
    reference: Iterable[Pair], predicted: Iterable[Pair], radius: int = 1
) -> Tuple[int, int, int]:
    """One-to-one shift matching; returns ``(tp, n_reference, n_predicted)``.

    Deterministic: candidate matches are sorted by ``(|di| + |dj|, di, dj, ...)``
    and greedily accepted.  On near-canonical structures this equals the maximum
    bipartite matching; ties are resolved lexicographically so reruns match.
    """
    refs = sorted(normalize_pairs(reference))
    preds = sorted(normalize_pairs(predicted))
    if not refs or not preds:
        return 0, len(refs), len(preds)
    candidates = []
    for pred_index, (i, j) in enumerate(preds):
        for ref_index, (ri, rj) in enumerate(refs):
            di = i - ri
            dj = j - rj
            if abs(di) <= radius and abs(dj) <= radius:
                candidates.append((abs(di) + abs(dj), di, dj, pred_index, ref_index))
    candidates.sort()
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    true_positive = 0
    for _distance, _di, _dj, pred_index, ref_index in candidates:
        if pred_index in used_pred or ref_index in used_ref:
            continue
        used_pred.add(pred_index)
        used_ref.add(ref_index)
        true_positive += 1
    return true_positive, len(refs), len(preds)


def flexible_prf(
    reference: Iterable[Pair], predicted: Iterable[Pair], radius: int = 1
) -> Triple:
    """Shift-``radius`` pair-level P/R/F1 with the same empty conventions."""
    refs = normalize_pairs(reference)
    preds = normalize_pairs(predicted)
    if not refs and not preds:
        return 1.0, 1.0, 1.0
    if not refs or not preds:
        return 0.0, 0.0, 0.0
    true_positive, n_ref, n_pred = flexible_matches(refs, preds, radius=radius)
    precision = true_positive / n_pred
    recall = true_positive / n_ref
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Official flexible metric (selection metric of the published protocol)
# ---------------------------------------------------------------------------
def _neighbourhood(pair: Pair, radius: int) -> List[Pair]:
    """Manhattan-ball support of the official ``_relax_ss`` dilation.

    Upstream adds four rolls (left/right/up/down), so a cell is "near" a pair if
    it differs by at most ``radius`` in **one** coordinate at a time (centre plus
    orthogonal steps), not the full Chebyshev square.
    """
    i, j = pair
    cells: List[Pair] = [(i, j)]
    for step in range(1, radius + 1):
        cells.extend([(i + step, j), (i - step, j), (i, j + step), (i, j - step)])
    return cells


def official_flexible_prf(
    reference: Iterable[Pair], predicted: Iterable[Pair], radius: int = 1
) -> Triple:
    """Exact replica of ``rinalmo.utils.sec_struct`` precision/recall/F1.

    The official matrix relaxation (``_relax_ss``) marks a cell as positive when
    *any* reference/prediction pair lies within ``radius`` in both coordinates,
    which is equivalent to a neighbourhood-set lookup:

    * precision = |prediction within radius of the target| / |prediction|
    * recall    = |target within radius of the prediction| / |target|

    Unlike :func:`flexible_prf` this is **not** one-to-one (a single target pair
    may explain several predictions and vice versa), exactly like the published
    metric.  ``tests/test_metrics.py`` checks the equivalence against the matrix
    implementation and, when importable, against upstream ``ss_f1``.
    """
    refs = normalize_pairs(reference)
    preds = normalize_pairs(predicted)
    if not refs and not preds:
        return 1.0, 1.0, 1.0
    if not refs or not preds:
        return 0.0, 0.0, 0.0
    true_positive_precision = sum(
        1 for pair in preds if any(neighbour in refs for neighbour in _neighbourhood(pair, radius))
    )
    true_positive_recall = sum(
        1 for pair in refs if any(neighbour in preds for neighbour in _neighbourhood(pair, radius))
    )
    precision = true_positive_precision / len(preds)
    recall = true_positive_recall / len(refs)
    if precision + recall < 1e-5:  # official EPSILON guard
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Accumulator (one object per split / threshold)
# ---------------------------------------------------------------------------
class MetricAccumulator:
    """Accumulate per-RNA exact + flexible metrics, then macro-average.

    ``add`` is deliberately cheap (no numpy) because training loops call it once
    per RNA per threshold on 10k+ RNAs.
    """

    def __init__(self) -> None:
        self.rows: List[dict] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add(
        self,
        identifier: str,
        family: Hashable,
        reference: Iterable[Pair],
        predicted: Iterable[Pair],
        *,
        length: int | None = None,
    ) -> dict:
        reference = normalize_pairs(reference)
        predicted = normalize_pairs(predicted)
        exact = pair_prf(reference, predicted)
        flexible = flexible_prf(reference, predicted)
        official_flexible = official_flexible_prf(reference, predicted)
        row = {
            "id": identifier,
            "family": family,
            "length": length,
            "n_true": len(reference),
            "n_pred": len(predicted),
            "tp": len(reference & predicted),
            "exact_precision": exact[0],
            "exact_recall": exact[1],
            "exact_f1": exact[2],
            "flexible_precision": flexible[0],
            "flexible_recall": flexible[1],
            "flexible_f1": flexible[2],
            "official_flexible_precision": official_flexible[0],
            "official_flexible_recall": official_flexible[1],
            "official_flexible_f1": official_flexible[2],
        }
        self.rows.append(row)
        return row

    def summary(self, excluded_families: Iterable[object] = EXCLUDED_FAMILIES) -> dict:
        exact = rna_macro([(r["exact_precision"], r["exact_recall"], r["exact_f1"]) for r in self.rows])
        flexible = rna_macro(
            [(r["flexible_precision"], r["flexible_recall"], r["flexible_f1"]) for r in self.rows]
        )
        official_flexible = rna_macro(
            [
                (
                    r["official_flexible_precision"],
                    r["official_flexible_recall"],
                    r["official_flexible_f1"],
                )
                for r in self.rows
            ]
        )
        return {
            "n_rna": len(self.rows),
            "rna_macro_exact_precision": exact[0],
            "rna_macro_exact_recall": exact[1],
            "rna_macro_exact_f1": exact[2],
            "rna_macro_flexible_precision": flexible[0],
            "rna_macro_flexible_recall": flexible[1],
            "rna_macro_flexible_f1": flexible[2],
            "rna_macro_official_flexible_precision": official_flexible[0],
            "rna_macro_official_flexible_recall": official_flexible[1],
            "rna_macro_official_flexible_f1": official_flexible[2],
            "family_macro_exact_f1": family_macro(
                ((r["family"], r["exact_f1"]) for r in self.rows), excluded=excluded_families
            ),
            "family_macro_flexible_f1": family_macro(
                ((r["family"], r["flexible_f1"]) for r in self.rows), excluded=excluded_families
            ),
            "family_macro_official_flexible_f1": family_macro(
                ((r["family"], r["official_flexible_f1"]) for r in self.rows),
                excluded=excluded_families,
            ),
            "families": _per_family_summary(self.rows, excluded_families),
            "empty_conventions": empty_convention_stats(
                ((r["n_true"] == 0, r["n_pred"] == 0) for r in self.rows)
            ),
            "mean_n_true": (sum(r["n_true"] for r in self.rows) / len(self.rows)) if self.rows else 0.0,
            "mean_n_pred": (sum(r["n_pred"] for r in self.rows) / len(self.rows)) if self.rows else 0.0,
        }


def _per_family_summary(
    rows: Sequence[Mapping], excluded: Iterable[object] = EXCLUDED_FAMILIES
) -> Dict[str, Dict[str, float]]:
    excluded = tuple(excluded)
    buckets: Dict[Hashable, List[Mapping]] = defaultdict(list)
    for row in rows:
        if row["family"] in excluded:
            continue
        buckets[row["family"]].append(row)
    return {
        str(family): {
            "count": len(values),
            "exact_f1": sum(v["exact_f1"] for v in values) / len(values),
            "flexible_f1": sum(v["flexible_f1"] for v in values) / len(values),
            "official_flexible_f1": sum(v["official_flexible_f1"] for v in values)
            / len(values),
        }
        for family, values in buckets.items()
    }
