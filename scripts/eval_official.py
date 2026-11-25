#!/usr/bin/env python
"""scripts/eval_official.py -- score official RiNALMo fine-tuned checkpoints.

Purpose: reproduce the published RiNALMo numbers **with our evaluation code**
before training anything.  Two metric conventions are reported side by side:

(a) ``official`` -- ``rinalmo.utils.sec_struct.ss_precision/ss_recall/ss_f1``
    with ``allow_flexible_pairings=True`` when the upstream package is importable
    (this is exactly what the RiNALMo paper reports); otherwise a byte-faithful
    local replica (``src.official``) is used and ``metric_source`` says so.
    Per-RNA values are macro-averaged over the split, as in the official trainer.

(b) ``ours`` -- exact set-based F1 (no relaxation) and our one-to-one shift-1
    flexible F1 (``src.metrics``), same RNA-macro convention.

Decoding is the official ``prob_mat_to_sec_struct`` (copy of the probability
matrix, canonical + sharp-loop masks, strict ``> threshold``, greedy cleaning by
descending probability).  We also decode with our own greedy decoder and count
disagreements as a sanity check (they must be zero).

Inputs: canonical JSONL splits (``--split-file`` / ``--config``) **or** the
official ArchiveII CT trees (``--ct-dir NAME=/path/to/archiveII/tRNA/test``).

Examples
--------
    # random tiny model, CPU, no weights needed (local test path)
    python scripts/eval_official.py --random-init --ct-dir tRNA=$CT/tRNA/test \
        --out runs/official_smoke

    # real Zenodo checkpoint on the official bpRNA test split
    python scripts/eval_official.py --checkpoint weights/rinalmo_giga_ss_bprna_ft.pt \
        --config configs/baseline_e2e.yaml --split TS0 --out experiments/E0_official
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as rna_data  # noqa: E402
from src import official as rna_official  # noqa: E402
from src.decode import decode as our_decode  # noqa: E402
from src.metrics import flexible_prf, pair_prf  # noqa: E402
from src.model import register_rinalmo_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import resolve_device, save_json, sha256_file  # noqa: E402


def _official_metric_functions():
    """Return upstream ``(precision, recall, f1, source)`` or the replica."""
    try:
        from rinalmo.utils.sec_struct import (  # type: ignore
            ss_f1,
            ss_precision,
            ss_recall,
        )

        return ss_precision, ss_recall, ss_f1, "rinalmo.utils.sec_struct"
    except Exception:
        return None, None, None, "src.official.replica"


def _official_decode_matrix(probabilities: np.ndarray, sequence: str, threshold: float):
    """Official decoder when importable, else the local replica (same semantics)."""
    try:
        from rinalmo.utils.sec_struct import prob_mat_to_sec_struct  # type: ignore

        return prob_mat_to_sec_struct(
            np.array(probabilities, copy=True), sequence, threshold
        )
    except Exception:
        return rna_official.official_decode_probs(probabilities, sequence, threshold)


def parse_pairs(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def resolve_splits(args: argparse.Namespace):
    splits: Dict[str, Tuple[str, Any]] = {}
    config = None
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        eval_specs = (config.get("data") or {}).get("eval") or {}
        for name in args.split or []:
            if name not in eval_specs:
                raise KeyError(f"split {name!r} not in {args.config} data.eval")
            splits[name] = ("jsonl", eval_specs[name])
    for value in args.split_file or []:
        splits.update({parse_pairs(value)[0]: ("jsonl", parse_pairs(value)[1])})
    for value in args.ct_dir or []:
        splits.update({parse_pairs(value)[0]: ("ct", parse_pairs(value)[1])})
    if not splits:
        raise SystemExit("no splits: use --config+--split, --split-file, or --ct-dir")
    return splits, config


@torch.no_grad()
def run_split(
    model: rna_official.OfficialSecStructModel,
    dataset: rna_data.RNAStructureDataset,
    *,
    split_name: str,
    threshold: float,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
    max_records: int | None,
) -> dict:
    metrics_source = _official_metric_functions()
    ss_precision, ss_recall, ss_f1, metric_source = metrics_source
    if max_records is not None and len(dataset) > max_records:
        dataset = rna_data.RNAStructureDataset(
            dataset.records[:max_records],
            alphabet=dataset.alphabet,
            channels=dataset.channels,
            thermo=dataset.thermo,
            min_sep=dataset.min_sep,
        )
    loader = rna_data.make_dataloader(
        dataset, batch_size=batch_size, shuffle=False, bucket_size=64, num_workers=0
    )
    rows: List[dict] = []
    decoder_mismatches = 0
    prediction_path = out_dir / f"predictions_{split_name}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in loader:
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].tolist()
            for row, length in enumerate(lengths):
                sample_tokens = tokens[row : row + 1, : length + 2]
                # flash-attn requires fp16/bf16 inputs; the official checkpoint is fp32,
                # so run its forward inside autocast exactly like the official FP16 inference.
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type == "cuda" else torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(sample_tokens).float()
                probs = torch.sigmoid(logits[0, :length, :length]).cpu().numpy()
                sequence = batch["sequences"][row]
                reference = rna_data.pair_matrix(length, batch["pairs"][row])
                predicted_matrix = _official_decode_matrix(probs, sequence, threshold)
                official_pairs = rna_official.matrix_to_pairs(predicted_matrix)
                our_pairs = our_decode(logits[0, :length, :length].cpu().numpy(), sequence, threshold)
                mismatch = sorted(official_pairs) != sorted(our_pairs)
                decoder_mismatches += int(mismatch)

                if ss_f1 is not None:
                    precision = float(
                        ss_precision(reference, predicted_matrix, allow_flexible_pairings=True)
                    )
                    recall = float(
                        ss_recall(reference, predicted_matrix, allow_flexible_pairings=True)
                    )
                    f1 = float(ss_f1(reference, predicted_matrix, allow_flexible_pairings=True))
                else:
                    precision, recall, f1 = rna_official.official_prf_from_matrices(
                        reference, predicted_matrix
                    )
                exact = pair_prf(batch["pairs"][row], official_pairs)
                flexible = flexible_prf(batch["pairs"][row], official_pairs)
                row_record = {
                    "id": batch["ids"][row],
                    "split": split_name,
                    "family": batch["families"][row],
                    "length": length,
                    "threshold": threshold,
                    "n_true": len(set(map(tuple, batch["pairs"][row]))),
                    "n_pred": len(official_pairs),
                    "tp": len(set(map(tuple, batch["pairs"][row])) & set(official_pairs)),
                    "fp": len(official_pairs)
                    - len(set(map(tuple, batch["pairs"][row])) & set(official_pairs)),
                    "fn": len(set(map(tuple, batch["pairs"][row])))
                    - len(set(map(tuple, batch["pairs"][row])) & set(official_pairs)),
                    "official_precision": precision,
                    "official_recall": recall,
                    "official_f1": f1,
                    "exact_f1": exact[2],
                    "flexible_f1": flexible[2],
                    "decoder_mismatch": mismatch,
                }
                rows.append(row_record)
                handle.write(json.dumps(row_record, ensure_ascii=False) + "\n")

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    families: Dict[str, List[float]] = {}
    for row in rows:
        family = str(row["family"])
        if family in ("", "unknown", "NA", "None"):
            continue
        families.setdefault(family, []).append(row["exact_f1"])
    summary = {
        "split": split_name,
        "n_rna": len(rows),
        "threshold": threshold,
        "metric_source": metric_source,
        "decoder_mismatches": decoder_mismatches,
        "official": {
            "precision_macro": mean("official_precision"),
            "recall_macro": mean("official_recall"),
            "f1_macro": mean("official_f1"),
        },
        "ours": {
            "exact_precision_macro": mean("exact_precision")
            if "exact_precision" in rows[0]
            else None,
            "exact_f1_macro": mean("exact_f1"),
            "flexible_f1_macro": mean("flexible_f1"),
        },
        "per_family_exact_f1": {
            family: float(np.mean(values)) for family, values in sorted(families.items())
        },
        "family_macro_exact_f1": float(
            np.mean([np.mean(values) for values in families.values()])
        )
        if families
        else 0.0,
        "mean_n_true": mean("n_true"),
        "mean_n_pred": mean("n_pred"),
        "predictions_path": str(prediction_path),
    }
    # exact precision is not stored per row; recompute from tp/fp cheaply
    if rows:
        summary["ours"]["exact_precision_macro"] = float(
            np.mean(
                [
                    (row["tp"] / row["n_pred"]) if row["n_pred"] else 0.0
                    for row in rows
                ]
            )
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=None, help="official Zenodo *_ft.pt checkpoint")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="tiny randomly initialised official model (local CPU test, no weights)",
    )
    parser.add_argument("--lm-config", default="giga", help="official lm config (mega|giga|micro)")
    parser.add_argument("--num-blocks", type=int, default=2, help="official pair-head resnet blocks")
    parser.add_argument("--config", default=None, help="training YAML to resolve data.eval specs")
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--split-file", action="append", default=[], help="NAME=JSONL_PATH")
    parser.add_argument("--ct-dir", action="append", default=[], help="NAME=/path/to/archiveII/<family>/test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="override checkpoint threshold")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--rinalmo-root",
        default=str(ROOT / "third_party" / "RiNALMo-main"),
        help="local RiNALMo checkout (server: installed package takes precedence)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.rinalmo_root, "rinalmo").is_dir():
        register_rinalmo_path(args.rinalmo_root)
    device = resolve_device(args.device)

    if not args.checkpoint and not args.random_init:
        raise SystemExit("provide --checkpoint <official .pt> or --random-init for a smoke run")

    if args.random_init:
        model = rna_official.build_tiny_official_model().to(device)
        checkpoint_threshold = 0.5
        threshold_source = "default(0.5, random-init)"
        checkpoint_info = {"path": None, "sha256": None, "load_report": None}
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
        model = rna_official.OfficialSecStructModel(
            lm_config=args.lm_config, num_blocks=int(args.num_blocks)
        ).to(device)
        load_report = model.load_official_checkpoint(checkpoint_path, strict=True)
        checkpoint_threshold = float(model.threshold)
        threshold_source = "checkpoint"
        checkpoint_info = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "load_report": load_report,
        }
    model.eval()
    threshold = float(args.threshold) if args.threshold is not None else checkpoint_threshold
    if args.threshold is not None:
        threshold_source = "override"

    splits, _config = resolve_splits(args)
    alphabet = rna_data.get_alphabet()
    results: Dict[str, Any] = {}
    started = time.time()
    for name, (kind, spec) in splits.items():
        if kind == "ct":
            records = rna_data.load_ct_records(
                Path(spec) if Path(spec).is_absolute() else ROOT / spec,
                limit=args.max_records,
            )
            dataset = rna_data.RNAStructureDataset(records, alphabet=alphabet)
        else:
            dataset = rna_data.build_dataset_for_spec(
                spec,
                root=ROOT,
                alphabet=alphabet,
                limit=args.max_records,
            )
        summary = run_split(
            model,
            dataset,
            split_name=name,
            threshold=threshold,
            device=device,
            batch_size=int(args.batch_size),
            out_dir=out_dir,
            max_records=args.max_records,
        )
        summary["dataset"] = {
            "kind": kind,
            "spec": spec if isinstance(spec, (str, list)) else dict(spec),
            "num_records": len(dataset),
        }
        results[name] = summary
        print(
            f"OFFICIAL {name} n={summary['n_rna']} "
            f"official_f1={summary['official']['f1_macro']:.6f} "
            f"exact_f1={summary['ours']['exact_f1_macro']:.6f} "
            f"flexible_f1={summary['ours']['flexible_f1_macro']:.6f} "
            f"decoder_mismatches={summary['decoder_mismatches']}",
            flush=True,
        )

    metrics = {
        "checkpoint": checkpoint_info,
        "lm_config": args.lm_config,
        "num_blocks": int(args.num_blocks),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "device": str(device),
        "splits": results,
        "elapsed_seconds": time.time() - started,
        "script_sha256": sha256_file(Path(__file__)),
    }
    save_json(out_dir / "metrics.json", metrics)
    print(json.dumps({"out": str(out_dir), "splits": list(results)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
