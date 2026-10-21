"""scripts/_common.py -- 训练/评估脚本共享工具（数据加载、种子、评估循环）"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decoder import candidate_pairs, decode, decode_multi
from src.metrics import family_macro, pair_prf, rna_macro
from src.utils import pair_set_0b, pairs_from_str, to_0based, to_1based  # noqa: F401

THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def load_manifest(root: Path = ROOT):
    man = {}
    with open(root / "data" / "processed" / "manifest.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            man[r["sample_id"]] = r
    return man


def load_split(root: Path = ROOT):
    return json.load(open(root / "data" / "processed" / "split_ids.json"))


def load_family(root=ROOT):
    man=load_manifest(root)
    return {r['source_id']:r.get('family_id','') or 'unknown' for r in man.values()}


def load_embeddings(root: Path, split: str, tag: str = ""):
    return torch.load(root / "embeddings" / f"{split}{tag}.pt", map_location="cpu")


def build_candidates(seq: str, eligible_0b):
    """V_x = 任务允许位置；y=1 当且仅当属于 eligible 参考配对。"""
    cands = candidate_pairs(seq)
    ref = set(eligible_0b)
    ci = torch.tensor([c[0] for c in cands], dtype=torch.long)
    cj = torch.tensor([c[1] for c in cands], dtype=torch.long)
    labels = torch.tensor([1.0 if c in ref else 0.0 for c in cands], dtype=torch.float32)
    return ci, cj, labels


def config_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def precompute(ids, man, emb, device="cuda", to_gpu=True):
    """预计算每 RNA 的 V_x 张量、参考配对集合；可选把 embedding 常驻 GPU。

    返回 (emb_map, cand_map, ref_map)。
    """
    cand_map, ref_map = {}, {}
    for sid in ids:
        r = man[sid]
        ref = pair_set_0b(pairs_from_str(r["eligible_pairs"]))
        ref_map[sid] = ref
        ci, cj, labels = build_candidates(r["sequence"], ref)
        if to_gpu and device == "cuda":
            ci, cj, labels = ci.to(device), cj.to(device), labels.to(device)
        cand_map[sid] = (ci, cj, labels)
    if to_gpu and device == "cuda":
        emb = {sid: emb[sid].to(device, dtype=torch.float32) for sid in ids}
    return emb, cand_map, ref_map


def rna_logits(head, emb, device="cuda"):
    e = emb.to(device, dtype=torch.float32)
    return head(e)


@torch.no_grad()
def eval_thresholds(head, emb_map, ids, man, fam_map=None,
                    thresholds=THRESHOLDS, device="cuda", progress=False):
    """对每个阈值返回 RNA-macro P/R/F1；另给 family-macro F1（按 RNA 所属 family 分组）。"""
    head.eval()
    per_thr = {t: {"prf": [], "fam": [], "flags": []} for t in thresholds}
    for k, sid in enumerate(ids, 1):
        r = man[sid]
        seq = r["sequence"]
        ref = pair_set_0b(pairs_from_str(r["eligible_pairs"]))
        emb = emb_map[sid]
        logits = rna_logits(head, emb, device)
        arr = logits.detach().to("cpu").float().numpy()
        fam = fam_map.get(sid.split(":", 1)[1], "NA") if fam_map else None
        preds = decode_multi(arr, seq, thresholds)
        for t in thresholds:
            pred = set(preds[t])
            p, rec, f1 = pair_prf(ref, pred)
            per_thr[t]["prf"].append((p, rec, f1))
            per_thr[t]["flags"].append((len(ref) == 0, len(pred) == 0))
            if fam_map:
                per_thr[t]["fam"].append((fam, f1))
        if progress and k % 200 == 0:
            print(f"    eval {k}/{len(ids)}", flush=True)

    out = {}
    for t in thresholds:
        p, r, f1 = rna_macro(per_thr[t]["prf"])
        fm = family_macro(per_thr[t]["fam"]) if fam_map else None
        out[t] = {
            "P": p, "R": r, "F1": f1,
            "family_macro_F1": fm,
            "n": len(per_thr[t]["prf"]),
            "empty_both": sum(1 for a, b in per_thr[t]["flags"] if a and b),
            "empty_only_ref": sum(1 for a, b in per_thr[t]["flags"] if a and not b),
            "empty_only_pred": sum(1 for a, b in per_thr[t]["flags"] if b and not a),
            "_per_rna": [(ids[i], per_thr[t]["prf"][i][2]) for i in range(len(ids))],
        }
    return out


def best_threshold(ev, key="F1"):
    """按 VL0 上 eligible RNA-macro F1 最大选阈值；并列取更小阈值（预定规则）。"""
    best_t, best_v = None, -1.0
    for t in sorted(ev.keys()):
        v = ev[t][key]
        if v is not None and v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, ensure_ascii=False)
