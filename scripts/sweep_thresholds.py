#!/usr/bin/env python
"""Diagnostic: sweep decoding thresholds on a test split for a trained checkpoint.

Computes the probability matrix once per RNA, then decodes at every threshold and
reports exact/flexible/official-flexible F1 (macro over RNAs). Used to separate
"bad model" from "bad threshold transfer" across families.

Usage (server):
  python scripts/sweep_thresholds.py --checkpoint runs/X/best.pt --config my-run.yaml \
      --split 5s --out runs/X/sweep_5s.json [--thresholds 0.01:0.40:0.01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataset_for_spec  # noqa: E402
from src.decode import decode_multi, threshold_grid  # noqa: E402
from src.metrics import flexible_prf, official_flexible_prf, pair_prf  # noqa: E402
from src.model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="0.01:0.40:0.01")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    model.eval()

    spec = (cfg.get("data", {}).get("eval") or {}).get(args.split)
    if spec is None:
        raise SystemExit(f"split {args.split!r} not in config data.eval")
    dataset = build_dataset_for_spec(
        spec,
        root=".",
        channels=cfg.get("data", {}).get("channels") or (),
        max_length=cfg.get("data", {}).get("max_length"),
        min_sep=cfg.get("data", {}).get("min_sep", 4),
    )
    n = len(dataset) if not args.max_records else min(len(dataset), args.max_records)
    print(f"records={n}", flush=True)

    thresholds = threshold_grid(args.thresholds)
    acc = {t: {"exact": [], "flexible": [], "official": [], "n_pred": 0} for t in thresholds}
    n_true = 0

    model.eval()
    with torch.no_grad():
        for index in range(n):
            item = dataset[index]
            tokens = item["tokens"].unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(tokens)
            probs = torch.sigmoid(logits.float())[0].cpu().numpy()
            seq = item["sequence"]
            true_pairs = [tuple(p) for p in item["pairs"]]
            n_true += len(true_pairs)
            decoded = decode_multi(probs, seq, thresholds)
            for t, pairs in zip(thresholds, decoded):
                acc[t]["exact"].append(pair_prf(true_pairs, pairs)[2])
                acc[t]["flexible"].append(flexible_prf(true_pairs, pairs)[2])
                acc[t]["official"].append(official_flexible_prf(true_pairs, pairs)[2])
                acc[t]["n_pred"] += len(pairs)
            if (index + 1) % 200 == 0:
                print(f"progress {index + 1}/{n}", flush=True)

    curve = []
    for t in thresholds:
        curve.append(dict(
            threshold=float(t),
            exact_f1=float(np.mean(acc[t]["exact"])),
            flexible_f1=float(np.mean(acc[t]["flexible"])),
            official_flexible_f1=float(np.mean(acc[t]["official"])),
            mean_pred_pairs=float(acc[t]["n_pred"] / n),
        ))
    best = max(curve, key=lambda row: row["flexible_f1"])
    best_exact = max(curve, key=lambda row: row["exact_f1"])
    out = dict(split=args.split, checkpoint=args.checkpoint, n_rna=n,
               mean_true_pairs=n_true / n, best_by_flexible=best, best_by_exact=best_exact, curve=curve)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(n=n, mean_true=n_true / n, best=best, best_exact=best_exact), indent=2))


if __name__ == "__main__":
    main()
