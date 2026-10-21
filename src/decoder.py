"""src/decoder.py -- 固定贪心解码器（计划 §15 / README §5.4）

允许配对: AU/UA/GC/CG/GU/UG, i<j, j-i>=4, 每 nt 至多一个 partner, 允许 crossing。
流程: sigmoid(logit) -> 过滤阈值以下 -> 按分数降序 -> 贪心接受（两端均未占用）。
相同分数按固定索引顺序处理，保证可重现。
"""
from __future__ import annotations

import numpy as np

from .utils import CANONICAL, MIN_SEP


def candidate_pairs(seq: str, min_sep: int = MIN_SEP):
    """返回所有合法候选 (i,j) 0-based 上三角。"""
    L = len(seq)
    out = []
    for i in range(L):
        si = seq[i]
        for j in range(i + min_sep, L):
            if (si, seq[j]) in CANONICAL:
                out.append((i, j))
    return out


def decode(scores: np.ndarray, seq: str, threshold: float, min_sep: int = MIN_SEP):
    """scores: (L,L) 对称 logits (numpy)。返回 0-based 配对列表 [(i,j), ...]。"""
    L = scores.shape[0]
    assert scores.shape == (L, L)
    probs = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
    cands = candidate_pairs(seq, min_sep)
    if not cands:
        return []
    ii = np.array([c[0] for c in cands])
    jj = np.array([c[1] for c in cands])
    # 对称分数取上三角 (i<j) 的值
    sc = probs[ii, jj]
    keep = sc > threshold
    ii, jj, sc = ii[keep], jj[keep], sc[keep]
    if len(ii) == 0:
        return []
    # 按分数降序；并列按 (i,j) 升序 -> 稳定可重现
    order = np.lexsort((jj, ii, -sc))
    ii, jj, sc = ii[order], jj[order], sc[order]
    used = np.zeros(L, dtype=bool)
    pairs = []
    for i, j in zip(ii, jj):
        if not used[i] and not used[j]:
            used[i] = True
            used[j] = True
            pairs.append((int(i), int(j)))
    return sorted(pairs)


def decode_from_logits_tensor(logits, seq, threshold, min_sep=MIN_SEP):
    """logits: torch (L,L) -> 0-based 配对列表。"""
    arr = logits.detach().to("cpu").float().numpy()
    return decode(arr, seq, threshold, min_sep)


def decode_multi(scores: np.ndarray, seq: str, thresholds, min_sep: int = MIN_SEP):
    """一次排序、多阈值解码（等价于逐阈值调用 decode，但快得多）。

    返回 {threshold: [(i,j), ...]}。
    """
    L = scores.shape[0]
    probs = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
    cands = candidate_pairs(seq, min_sep)
    out = {t: [] for t in thresholds}
    if not cands:
        return out
    ii = np.array([c[0] for c in cands])
    jj = np.array([c[1] for c in cands])
    sc = probs[ii, jj]
    order = np.lexsort((jj, ii, -sc))
    ii, jj, sc = ii[order], jj[order], sc[order]
    for t in thresholds:
        used = np.zeros(L, dtype=bool)
        pairs = []
        for a, b, s in zip(ii, jj, sc):
            if s <= t:
                break
            if not used[a] and not used[b]:
                used[a] = True
                used[b] = True
                pairs.append((int(a), int(b)))
        out[t] = sorted(pairs)
    return out
