"""src/stem_candidates.py -- 正 stem、竞争负 stem 的枚举与匹配（计划 §17-§18, README §6.2-§6.4）

连续 stem: H(i,j,m) = {(i+t, j-t): t=0..m-1}（0-based, i<j）

关键递推（反对角线）:
    run[i,j] = legal[i,j] ? 1 + run[i+1,j-1] : 0
run[i,j] = 以 (i,j) 为外层配对、向内连续合法的最大 stem 长度。
"""
from __future__ import annotations

import numpy as np

from .utils import (CANONICAL, MIN_SEP, SPAN_BINS, gc_count, gu_count,
                    span_bin, stem_pairs)


# --------------------------------------------------------------------------
# 反对角线递推
# --------------------------------------------------------------------------
def run_lengths(legal: np.ndarray) -> np.ndarray:
    """legal: (L,L) bool（仅上三角有意义）-> run: (L,L) int16"""
    L = legal.shape[0]
    run = np.zeros((L, L), dtype=np.int16)
    for d in range(2, 2 * L - 1):
        i_lo = max(0, d - (L - 1))
        i_hi = min((d - 1) // 2, L - 1)
        if i_hi < i_lo:
            continue
        ii = np.arange(i_hi, i_lo - 1, -1)   # 由内向外
        jj = d - ii
        vals = legal[ii, jj]
        k = np.arange(len(ii))
        last_false = np.maximum.accumulate(np.where(~vals, k, -1))
        run[ii, jj] = np.where(vals, k - last_false, 0)
    return run


def legal_matrix(seq: str, min_sep: int = MIN_SEP) -> np.ndarray:
    L = len(seq)
    m = np.zeros((L, L), dtype=bool)
    for i in range(L):
        si = seq[i]
        for j in range(i + min_sep, L):
            if (si, seq[j]) in CANONICAL:
                m[i, j] = True
    return m


# --------------------------------------------------------------------------
# 正 stem
# --------------------------------------------------------------------------
def maximal_positive_stems(seq: str, eligible_0b, min_stem: int = 3):
    """返回 maximal 正 stem 列表 [(i, j, M)]，M = 连续长度（>=min_stem）。"""
    L = len(seq)
    legal = np.zeros((L, L), dtype=bool)
    for i, j in eligible_0b:
        legal[i, j] = True
    run = run_lengths(legal)
    out = []
    for i, j in eligible_0b:
        M = int(run[i, j])
        if M < min_stem:
            continue
        # maximal: 不能再向外扩展
        if i - 1 >= 0 and j + 1 < L and legal[i - 1, j + 1]:
            continue
        out.append((i, j, M))
    return out


def sample_positive_anchors(seq, eligible_0b, k: int, rng, stem_lengths=(3, 4, 5, 6)):
    """每 RNA 每 epoch 采样 <=k 个正 anchor。
    先均匀选 maximal stem，再在其内部选长度与窗口（避免长 stem 被过度加权）。
    返回 [(i, j, m), ...]
    """
    maximal = maximal_positive_stems(seq, eligible_0b, min_stem=min(stem_lengths))
    if not maximal:
        return []
    n = min(k, len(maximal))
    idxs = rng.sample(range(len(maximal)), n)
    anchors = []
    for t in idxs:
        i, j, M = maximal[t]
        m_hi = min(max(stem_lengths), M)
        m_lo = min(stem_lengths)
        if m_hi < m_lo:
            continue
        m = rng.randint(m_lo, m_hi)
        off = rng.randint(0, M - m)
        anchors.append((i + off, j - off, m))
    return anchors


# --------------------------------------------------------------------------
# 竞争负 stem
# --------------------------------------------------------------------------
def enumerate_negative_stems(seq: str, ref_set_0b, stem_lengths=(3, 4, 5, 6)):
    """枚举所有长度 3-6、全部 label=0 的连续合法 stem。
    返回 (list[(i,j,m)], dict nt -> list of (i,j,m))
    """
    L = len(seq)
    legal = legal_matrix(seq)
    if ref_set_0b:
        for i, j in ref_set_0b:
            legal[i, j] = False          # 负 stem 的每个 pair 必须 label=0
    run = run_lengths(legal)
    stems = []
    by_nt = {}
    m_lo, m_hi = min(stem_lengths), max(stem_lengths)
    for i in range(L):
        row = run[i]
        for j in range(i + 2 * m_lo - 2, L):     # 最内层也需满足 min_sep
            R = int(row[j])
            if R < m_lo:
                continue
            for m in range(m_lo, min(m_hi, R) + 1):
                stems.append((i, j, m))
                for t in range(m):
                    a, b = i + t, j - t
                    by_nt.setdefault(a, []).append((i, j, m))
                    by_nt.setdefault(b, []).append((i, j, m))
    return stems, by_nt


def find_competing_negatives(pos, seq, ref_set_0b, by_nt,
                             max_gc_diff: int = 1, max_gu_diff: int = 1,
                             same_span_bin: bool = True, max_candidates=None):
    """对正 anchor pos=(i,j,m) 找满足 §18 全部条件的竞争负 stem。"""
    pi, pj, m = pos
    p_pairs = stem_pairs(pi, pj, m)
    p_nts = set()
    p_partner = {}
    for a, b in p_pairs:
        p_nts.add(a); p_partner[a] = b
        p_nts.add(b); p_partner[b] = a
    p_gc = gc_count(seq, p_pairs)
    p_gu = gu_count(seq, p_pairs)
    p_span_bin = span_bin(pj - pi)

    seen = set()
    out = []
    for nt in p_nts:
        for cand in by_nt.get(nt, ()):
            ci, cj, cm = cand
            if cm != m:
                continue
            if cand in seen:
                continue
            seen.add(cand)
            # 条件 4: 全部 label=0（枚举时已保证）
            # 条件 5/6: 至少共享一个 nt，且该 nt 的 partner 不同
            c_pairs = stem_pairs(ci, cj, m)
            c_partner = {}
            c_nts = set()
            for a, b in c_pairs:
                c_nts.add(a); c_partner[a] = b
                c_nts.add(b); c_partner[b] = a
            shared = p_nts & c_nts
            if not shared:
                continue
            if not any(p_partner[x] != c_partner[x] for x in shared):
                continue
            # 条件 7/8: GC / GU 数量差
            if abs(p_gc - gc_count(seq, c_pairs)) > max_gc_diff:
                continue
            if abs(p_gu - gu_count(seq, c_pairs)) > max_gu_diff:
                continue
            # 条件 9: 同一 span bin
            if same_span_bin:
                cb = span_bin(cj - ci)
                if cb != p_span_bin or cb < 0:
                    continue
            out.append(cand)
            if max_candidates and len(out) >= max_candidates:
                return out
    return out


def diagnose_filter_reasons(pos, seq, ref_set_0b, by_nt,
                            max_gc_diff=1, max_gu_diff=1, same_span_bin=True):
    """诊断：对每个候选统计被哪一条限制过滤（用于 §24 LOW_COVERAGE 分析）。"""
    pi, pj, m = pos
    p_pairs = stem_pairs(pi, pj, m)
    p_nts, p_partner = set(), {}
    for a, b in p_pairs:
        p_nts.add(a); p_partner[a] = b
        p_nts.add(b); p_partner[b] = a
    p_gc = gc_count(seq, p_pairs)
    p_gu = gu_count(seq, p_pairs)
    p_span_bin = span_bin(pj - pi)

    reasons = {"total": 0, "wrong_length": 0, "no_shared_nt": 0,
               "same_partner": 0, "gc": 0, "gu": 0, "span": 0, "ok": 0}
    seen = set()
    for nt in p_nts:
        for cand in by_nt.get(nt, ()):
            if cand in seen:
                continue
            seen.add(cand)
            reasons["total"] += 1
            ci, cj, cm = cand
            if cm != m:
                reasons["wrong_length"] += 1
                continue
            c_pairs = stem_pairs(ci, cj, m)
            c_partner = {}
            for a, b in c_pairs:
                c_partner[a] = b; c_partner[b] = a
            shared = p_nts & set(c_partner)
            if not shared:
                reasons["no_shared_nt"] += 1
                continue
            if not any(p_partner[x] != c_partner[x] for x in shared):
                reasons["same_partner"] += 1
                continue
            if abs(p_gc - gc_count(seq, c_pairs)) > max_gc_diff:
                reasons["gc"] += 1
                continue
            if abs(p_gu - gu_count(seq, c_pairs)) > max_gu_diff:
                reasons["gu"] += 1
                continue
            if same_span_bin and span_bin(cj - ci) != p_span_bin:
                reasons["span"] += 1
                continue
            reasons["ok"] += 1
    return reasons


# --------------------------------------------------------------------------
# 全部正 anchor 枚举 + 预计算索引（供 mine_negatives.py 高效使用）
# --------------------------------------------------------------------------
def enumerate_positive_anchors(seq, eligible_0b, stem_lengths=(3, 4, 5, 6)):
    """枚举全部 (maximal stem, 内部窗口) 正 anchor -> [(i,j,m), ...]。"""
    out = []
    m_lo, m_hi = min(stem_lengths), max(stem_lengths)
    for (i, j, M) in maximal_positive_stems(seq, eligible_0b, min_stem=m_lo):
        for m in range(m_lo, min(m_hi, M) + 1):
            for off in range(0, M - m + 1):
                out.append((i + off, j - off, m))
    return out


class StemInfo:
    """一个 stem 的预计算信息。"""
    __slots__ = ("key", "pairs", "nts", "partner", "gc", "gu", "sbin")

    def __init__(self, seq, i, j, m):
        self.key = (i, j, m)
        self.pairs = stem_pairs(i, j, m)
        self.nts = set()
        self.partner = {}
        for a, b in self.pairs:
            self.nts.add(a); self.partner[a] = b
            self.nts.add(b); self.partner[b] = a
        self.gc = gc_count(seq, self.pairs)
        self.gu = gu_count(seq, self.pairs)
        self.sbin = span_bin(j - i)


class NegativeStemIndex:
    """预计算全部合法全负 stem 的索引，支持快速竞争负例查找与打分。"""

    def __init__(self, seq, ref_set_0b, stem_lengths=(3, 4, 5, 6)):
        self.seq = seq
        self.stems, by_nt = enumerate_negative_stems(seq, ref_set_0b, stem_lengths)
        self.info = {}
        for (i, j, m) in self.stems:
            self.info[(i, j, m)] = StemInfo(seq, i, j, m)
        self.by_nt_keys = {}
        for nt, lst in by_nt.items():
            self.by_nt_keys[nt] = list(dict.fromkeys(lst))

    def find(self, pos, max_gc_diff=1, max_gu_diff=1, same_span_bin=True,
             max_candidates=None):
        pi, pj, m = pos
        pinfo = StemInfo(self.seq, pi, pj, m)
        seen = set()
        out = []
        for nt in pinfo.nts:
            for key in self.by_nt_keys.get(nt, ()):
                if key in seen:
                    continue
                seen.add(key)
                ci, cj, cm = key
                if cm != m:
                    continue
                cinfo = self.info[key]
                shared = pinfo.nts & cinfo.nts
                if not shared:
                    continue
                if not any(pinfo.partner[x] != cinfo.partner[x] for x in shared):
                    continue
                if abs(pinfo.gc - cinfo.gc) > max_gc_diff:
                    continue
                if abs(pinfo.gu - cinfo.gu) > max_gu_diff:
                    continue
                if same_span_bin and cinfo.sbin != pinfo.sbin:
                    continue
                out.append(key)
                if max_candidates and len(out) >= max_candidates:
                    return out
        return out

    def scores(self, logits_np):
        """{stem_key: 平均 logit}"""
        out = {}
        for key, info in self.info.items():
            s = 0.0
            for (a, b) in info.pairs:
                s += float(logits_np[a, b])
            out[key] = s / len(info.pairs)
        return out
