"""src/metrics.py -- 配对级评价指标（计划 §27/§35, README §10.1）

主指标: exact base-pair RNA-macro F1
  * 每条 RNA 独立算 F1（0-based 或 1-based 均可，只要 ref/pred 同口径），再对 RNA 平均。
  * 约定: ref 与 pred 均空 -> P=R=F1=1；仅一方为空 -> P=R=F1=0。
另报告 precision / recall，以及 family-macro F1（先 RNA 内、再 family 内、再 family 间）。
"""
from __future__ import annotations

from collections import defaultdict


def pair_prf(ref, pred):
    """ref/pred: 可哈希配对集合（统一 i<j）。返回 (p, r, f1)。"""
    ref = set(ref)
    pred = set(pred)
    if not ref and not pred:
        return 1.0, 1.0, 1.0
    if not ref or not pred:
        return 0.0, 0.0, 0.0
    tp = len(ref & pred)
    p = tp / len(pred)
    r = tp / len(ref)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


def rna_macro(per_rna):
    """per_rna: list of (p, r, f1) -> 均值三元组。"""
    if not per_rna:
        return 0.0, 0.0, 0.0
    n = len(per_rna)
    return (sum(x[0] for x in per_rna) / n,
            sum(x[1] for x in per_rna) / n,
            sum(x[2] for x in per_rna) / n)


def family_macro(per_rna_with_family):
    """per_rna_with_family: list of (family, f1) -> family 内平均再 family 间平均。"""
    buckets = defaultdict(list)
    for fam, f1 in per_rna_with_family:
        if fam not in (None,"unknown","NA",""): buckets[fam].append(f1)
    if not buckets:
        return 0.0
    fam_means = [sum(v) / len(v) for v in buckets.values()]
    return sum(fam_means) / len(fam_means)


def empty_convention_stats(per_rna_flags):
    """per_rna_flags: list of (ref_empty, pred_empty) -> 计数。"""
    both = sum(1 for a, b in per_rna_flags if a and b)
    only_ref = sum(1 for a, b in per_rna_flags if a and not b)
    only_pred = sum(1 for a, b in per_rna_flags if b and not a)
    return {"both_empty": both, "only_ref_empty": only_ref, "only_pred_empty": only_pred}
