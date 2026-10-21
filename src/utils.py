"""src/utils.py -- 共享工具：配对规则、跨度分箱、bpseq/字符串解析。

索引约定（README §2.1）:
  * 文件 / 导出（CT、pair list、manifest）使用 **1-based**
  * 张量与内部计算使用 **0-based**
  转换只发生在导入/导出边界。
"""
from __future__ import annotations

CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
MIN_SEP = 4
SPAN_BINS = [(4, 31), (32, 63), (64, 127), (128, 255), (256, 511)]
STEM_LENGTHS = [3, 4, 5, 6]


def is_legal_pair(seq: str, i: int, j: int, min_sep: int = MIN_SEP) -> bool:
    """0-based 合法配对：i<j、j-i>=min_sep、AU/UA/GC/CG/GU/UG。"""
    return (j - i) >= min_sep and (seq[i], seq[j]) in CANONICAL


def span_bin(span: int):
    """外层配对跨度 -> 分箱索引；不在任何箱内返回 -1。"""
    for k, (lo, hi) in enumerate(SPAN_BINS):
        if lo <= span <= hi:
            return k
    return -1


def pairs_from_str(s: str):
    """manifest 中的 '1-118;2-117' -> [(1,118), ...]（保持 1-based）。"""
    if not s:
        return []
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split("-")
        out.append((int(a), int(b)))
    return out


def to_0based(pairs_1b):
    return [(i - 1, j - 1) for i, j in pairs_1b]


def to_1based(pairs_0b):
    return [(i + 1, j + 1) for i, j in pairs_0b]


def pair_set_0b(pairs_1b):
    return {(i - 1, j - 1) for i, j in pairs_1b}


def gc_count(seq: str, pairs) -> int:
    return sum(1 for i, j in pairs if seq[i] in "GC" and seq[j] in "GC")


def gu_count(seq: str, pairs) -> int:
    return sum(1 for i, j in pairs if {seq[i], seq[j]} == {"G", "U"})


def stem_pairs(i: int, j: int, m: int):
    """H(i,j,m) = {(i+t, j-t): t=0..m-1}（0-based，i<j）。"""
    return [(i + t, j - t) for t in range(m)]


def stem_span(i: int, j: int) -> int:
    return j - i


# --------------------------------------------------------------------------
# CT 读写（1-based，CT 为权威输出格式；扩展 dot-bracket 仅供展示）
# --------------------------------------------------------------------------
def ct_from_pairs(seq: str, pairs_1b, header: str = "RNA"):
    """生成 CT 行列表。pairs_1b: [(i,j)] 1-based，i<j。"""
    partner = [0] * (len(seq) + 1)
    for i, j in pairs_1b:
        partner[i] = j
        partner[j] = i
    lines = [f"{len(seq)}\t{header}"]
    for k in range(1, len(seq) + 1):
        lines.append(f"{k}\t{seq[k-1]}\t{k-1 if k > 1 else 0}\t"
                     f"{k+1 if k < len(seq) else 0}\t{partner[k]}\t{k}")
    return lines


def pairs_from_ct(lines):
    """从 CT 行解析 1-based 配对集合（i<j）。"""
    data = [ln for ln in lines if ln.strip()]
    header = data[0]
    L = int(header.split()[0])
    pairs = set()
    for ln in data[1:]:
        parts = ln.split()
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        partner = int(parts[4])
        if partner and partner != 0:
            pairs.add((min(idx, partner), max(idx, partner)))
    assert L == len(data) - 1, f"CT length mismatch: header {L} vs {len(data)-1} rows"
    return sorted(pairs)
