#!/usr/bin/env python
"""scripts/evaluate.py -- frozen checkpoint -> predictions.jsonl + metrics.json.

Any split, any source:
  * ``--config configs/baseline_e2e.yaml --split TS0`` reuses the config's
    ``data.eval`` spec (the normal path for the bpRNA benchmarks);
  * ``--split-file NAME=/abs/or/relative/file.jsonl.gz`` scores one explicit file;
  * ``--ct-dir NAME=/path/to/archiveII/<family>/test`` scores an official CT
    directory (ArchiveII family-fold protocol).

The checkpoint's threshold is used by default: test-time threshold tuning is
forbidden by the protocol.  ``--threshold`` exists for diagnostics and is
recorded as ``threshold_source: "override"`` in ``metrics.json``.

Reverse-complement TTA is opt-in and recorded in ``metrics.json``::

    python scripts/evaluate.py --checkpoint runs/X/best.pt --config configs/baseline_e2e.yaml \
        --split TS0 --tta rc --tta-weight 0.5

The RC view halves are mixed in probability space with weight ``--tta-weight``
(default 0.5) and decoded with the same greedy rule; ``--tta-weight 0`` is
exactly the non-TTA path.  ``--tta none`` (default) is unchanged.

Outputs (under ``--out``, default ``<checkpoint dir>/eval``):
  metrics.json            per-split macro metrics + families + provenance + tta
  predictions_<split>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as rna_data  # noqa: E402
from src.decode import decode_multi  # noqa: E402
from src.metrics import MetricAccumulator  # noqa: E402
from src.model import build_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (  # noqa: E402
    autocast_context,
    batch_to_device,
    resolve_device,
    save_json,
    sha256_file,
)


# ---------------------------------------------------------------------------
# Reverse-complement test-time augmentation (opt-in via ``--tta rc``)
# ---------------------------------------------------------------------------
# RNA complement, including the IUPAC codes present in the official manifests.
# Unknown characters become ``N`` so the RC view never crashes on odd input.
RC_COMPLEMENT = {
    "A": "U",
    "U": "A",
    "T": "A",
    "C": "G",
    "G": "C",
    "I": "C",
    "R": "Y",
    "Y": "R",
    "K": "M",
    "M": "K",
    "B": "V",
    "V": "B",
    "D": "H",
    "H": "D",
    "S": "S",
    "W": "W",
    "N": "N",
    "-": "-",
}


def reverse_complement_sequence(sequence: str) -> str:
    """Reverse complement of an RNA sequence (T tolerated, unknown bases -> N)."""
    normalised = sequence.strip().upper().replace("T", "U")
    return "".join(RC_COMPLEMENT.get(base, "N") for base in reversed(normalised))


def rc_matrix_to_original(matrix: torch.Tensor) -> torch.Tensor:
    """Map an RC-coordinate pair matrix back to original coordinates.

    RC index ``k`` corresponds to original ``n-1-k``, so an RC pair ``(i, j)``
    is the original pair ``(n-1-j, n-1-i)`` (still ``i' < j'``).  On a matrix
    this is a flip of both axes followed by a transpose.  The operation is an
    involution: applying it twice returns the input.
    """
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"pair matrix must be square, got {tuple(matrix.shape)}")
    return matrix.flip(-1).flip(-2).transpose(-1, -2)


def build_rc_tokens(sequences: Sequence[str], alphabet, pad_to_len: int) -> torch.Tensor:
    encoded = [
        alphabet.encode(reverse_complement_sequence(sequence), pad_to_len=pad_to_len)
        for sequence in sequences
    ]
    return torch.tensor(encoded, dtype=torch.long)


def transform_channels_for_rc(
    channels: torch.Tensor | None, channel_names: Sequence[str]
) -> torch.Tensor | None:
    """Transform per-pair feature channels into RC coordinates.

    ``bpp``/``stack``/``mfe``/``mea``/``centroid`` are sequence-specific pair
    matrices and must be remapped with the same index transform; ``distance``
    (|i-j|) and ``mask`` (candidate geometry) are invariant under the RC index
    map and are left untouched.
    """
    if channels is None:
        return None
    transformed = channels.clone()
    for index, name in enumerate(channel_names):
        if name in ("bpp", "stack", "mfe", "mea", "centroid"):
            transformed[:, index] = rc_matrix_to_original(transformed[:, index])
    return transformed


def build_rc_batch(
    batch: Dict[str, Any], alphabet, channel_names: Sequence[str]
) -> Dict[str, Any]:
    """Model-agnostic RC batch: same model call, RC sequences and RC channels."""
    rc_batch = dict(batch)
    rc_batch["tokens"] = build_rc_tokens(
        batch["sequences"], alphabet, pad_to_len=batch["tokens"].shape[1]
    ).to(batch["tokens"].device)
    if "channels" in batch:
        rc_batch["channels"] = transform_channels_for_rc(batch["channels"], channel_names)
    return rc_batch


def forward_logits(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    with autocast_context(device, amp):
        return model(
            batch["tokens"],
            lengths=batch["lengths"],
            padding_mask=batch["padding_mask"],
            channels=batch.get("channels"),
            pair_mask=batch["pair_mask"],
            channel_names=model.data_channels(),
        )


def tta_scores(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    tta_mode: str,
    tta_weight: float,
    alphabet,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Return decoder-ready scores and a small TTA provenance dict.

    ``none`` returns the raw logits (identical to the pre-TTA behaviour).  ``rc``
    averages ``(1-w) * sigmoid(logits) + w * sigmoid(rc_logits_mapped)`` and
    converts the mixture back to logit space so the existing greedy decoder is
    used unchanged.  ``w == 0`` short-circuits to the raw logits, so weight 0 is
    *exactly* the non-TTA path (unit-tested).
    """
    logits = forward_logits(model, batch, device, amp)
    if tta_mode != "rc":
        return logits, {"mode": "none", "weight": None, "views": 1}
    weight = float(tta_weight)
    if weight == 0.0:
        return logits, {"mode": "rc", "weight": 0.0, "views": 1}
    rc_batch = build_rc_batch(batch, alphabet, model.data_channels())
    rc_logits = forward_logits(model, rc_batch, device, amp)
    rc_logits = rc_matrix_to_original(rc_logits.float())
    original_probabilities = torch.sigmoid(logits.float())
    rc_probabilities = torch.sigmoid(rc_logits)
    probabilities = (1.0 - weight) * original_probabilities + weight * rc_probabilities
    probabilities = probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
    scores = torch.log(probabilities) - torch.log1p(-probabilities)
    return scores, {"mode": "rc", "weight": weight, "views": 2}


def parse_split_file(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--split-file expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def resolve_splits(args: argparse.Namespace) -> Dict[str, Any]:
    splits: Dict[str, Any] = {}
    config = None
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        eval_specs = (config.get("data") or {}).get("eval") or {}
        for name in args.split or []:
            if name not in eval_specs:
                raise KeyError(
                    f"split {name!r} not in {args.config} data.eval (have {sorted(eval_specs)})"
                )
            splits[name] = ("jsonl", eval_specs[name])
    for value in args.split_file or []:
        name, path = parse_split_file(value)
        splits[name] = ("jsonl", path)
    for value in args.ct_dir or []:
        name, path = parse_split_file(value)
        splits[name] = ("ct", path)
    if not splits:
        raise SystemExit(
            "no splits requested: use --config+--split, --split-file NAME=PATH, or --ct-dir NAME=PATH"
        )
    return splits, config


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise TypeError(f"{path}: not a training checkpoint (missing 'model')")
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise ValueError("checkpoint has no embedded 'cfg.model'; cannot rebuild the architecture")
    model_cfg = copy.deepcopy(cfg["model"])
    # A self-contained training checkpoint already carries every backbone
    # weight; the pretrained file is only needed when the checkpoint is a
    # partial/head-only payload.  Opt-in skip lets us evaluate archived
    # checkpoints whose original pretrained path no longer exists without
    # touching the checkpoint itself.  Weights are still loaded strictly below.
    if os.environ.get("RNA_EVAL_SKIP_PRETRAINED") == "1":
        backbone = model_cfg.get("backbone") or {}
        if str(backbone.get("type", "rinalmo")).lower().startswith("rinalmo"):
            backbone["pretrained_path"] = None
            backbone["allow_random_init"] = True
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate_split_spec(
    model: torch.nn.Module,
    dataset: rna_data.RNAStructureDataset,
    *,
    threshold: float,
    device: torch.device,
    amp: bool,
    batch_size: int,
    out_dir: Path,
    split_name: str,
    tta_mode: str = "none",
    tta_weight: float = 0.5,
) -> dict:
    loader = rna_data.make_dataloader(
        dataset, batch_size=batch_size, shuffle=False, bucket_size=64, num_workers=0
    )
    accumulator = MetricAccumulator()
    prediction_path = out_dir / f"predictions_{split_name}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for batch in loader:
            batch = batch_to_device(batch, device)
            scores, tta_meta = tta_scores(
                model,
                batch,
                tta_mode=tta_mode,
                tta_weight=tta_weight,
                alphabet=dataset.alphabet,
                device=device,
                amp=amp,
            )
            logits = scores.float().cpu().numpy()
            for row, length in enumerate(batch["lengths"].tolist()):
                sequence = batch["sequences"][row]
                pairs = decode_multi(logits[row, :length, :length], sequence, [threshold])[
                    float(threshold)
                ]
                row_metrics = accumulator.add(
                    batch["ids"][row],
                    batch["families"][row],
                    batch["pairs"][row],
                    pairs,
                    length=length,
                )
                handle.write(
                    json.dumps(
                        {
                            "id": row_metrics["id"],
                            "split": split_name,
                            "family": row_metrics["family"],
                            "length": length,
                            "threshold": threshold,
                            "n_true": row_metrics["n_true"],
                            "n_pred": row_metrics["n_pred"],
                            "tp": row_metrics["tp"],
                            "fp": row_metrics["n_pred"] - row_metrics["tp"],
                            "fn": row_metrics["n_true"] - row_metrics["tp"],
                            "exact_precision": row_metrics["exact_precision"],
                            "exact_recall": row_metrics["exact_recall"],
                            "exact_f1": row_metrics["exact_f1"],
                            "flexible_precision": row_metrics["flexible_precision"],
                            "flexible_recall": row_metrics["flexible_recall"],
                            "flexible_f1": row_metrics["flexible_f1"],
                            "tta": tta_mode,
                            "tta_weight": tta_weight if tta_mode == "rc" else None,
                            "predicted_pairs": [list(pair) for pair in pairs],
                            "reference_pairs": [list(pair) for pair in batch["pairs"][row]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    summary = accumulator.summary()
    summary["predictions_path"] = str(prediction_path)
    summary["tta"] = {
        "mode": tta_mode,
        "weight": tta_weight if tta_mode == "rc" else None,
        "views": 2 if (tta_mode == "rc" and tta_weight > 0.0) else 1,
        "aggregation": "mean_probability" if tta_mode == "rc" else "none",
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="best.pt / last.pt / epochN.pt")
    parser.add_argument("--config", default=None, help="training YAML (to resolve data.eval specs)")
    parser.add_argument("--split", action="append", default=[], help="eval split name from the config")
    parser.add_argument(
        "--split-file", action="append", default=[], help="explicit JSONL: NAME=PATH (glob allowed)"
    )
    parser.add_argument(
        "--ct-dir", action="append", default=[], help="official CT directory: NAME=/path/to/test"
    )
    parser.add_argument("--out", default=None, help="output dir (default <checkpoint dir>/eval)")
    parser.add_argument("--threshold", type=float, default=None, help="diagnostic override only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--thermo-dir", default=None, help="override data.thermo_dir")
    parser.add_argument(
        "--tta",
        choices=["none", "rc"],
        default="none",
        help="test-time augmentation; default none (opt-in, recorded in metrics.json)",
    )
    parser.add_argument(
        "--tta-weight",
        type=float,
        default=0.5,
        help="weight of the reverse-complement view in [0, 1]; 0 reproduces non-TTA exactly",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= float(args.tta_weight) <= 1.0:
        raise SystemExit("--tta-weight must be inside [0, 1]")

    checkpoint_path = Path(args.checkpoint).resolve()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, cfg = build_model_from_checkpoint(checkpoint, device)
    frozen_threshold = float(checkpoint.get("threshold", 0.5))
    threshold = float(args.threshold) if args.threshold is not None else frozen_threshold
    threshold_source = "override" if args.threshold is not None else "checkpoint"
    out_dir = Path(args.out) if args.out else checkpoint_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits, config = resolve_splits(args)
    data_cfg = (cfg.get("data") or {})
    channels = list(model.data_channels())
    thermo_dir = args.thermo_dir or data_cfg.get("thermo_dir")
    thermo = None
    if any(channel in rna_data.THERMO_CHANNELS for channel in channels):
        if not thermo_dir:
            raise SystemExit(
                f"model needs channels {channels} but no thermo dir in checkpoint cfg or --thermo-dir"
            )
        directory = Path(thermo_dir)
        if not directory.is_absolute():
            directory = ROOT / directory
        thermo = rna_data.ThermoFeatureStore(
            directory, channels=tuple(c for c in channels if c in rna_data.THERMO_CHANNELS)
        )
    alphabet = rna_data.get_alphabet(bool(data_cfg.get("use_rinalmo_alphabet", True)))
    amp = device.type == "cuda"

    results: Dict[str, Any] = {}
    started = time.time()
    for name, (kind, spec) in splits.items():
        if kind == "ct":
            records = rna_data.load_ct_records(
                Path(spec) if Path(spec).is_absolute() else ROOT / spec,
                limit=args.max_records,
            )
            dataset = rna_data.RNAStructureDataset(
                records,
                alphabet=alphabet,
                channels=channels,
                thermo=thermo,
                min_sep=int(data_cfg.get("min_sep", rna_data.MIN_SEP)),
            )
        else:
            limits = args.max_records if args.max_records else data_cfg.get("max_records")
            dataset = rna_data.build_dataset_for_spec(
                spec,
                root=ROOT,
                channels=channels,
                thermo=thermo,
                alphabet=alphabet,
                limit=limits,
                min_sep=int(data_cfg.get("min_sep", rna_data.MIN_SEP)),
                max_length=data_cfg.get("max_length"),
                strict_sequence=bool(data_cfg.get("strict_sequence", False)),
                strict_pairs=bool(data_cfg.get("strict_pairs", False)),
                require_eligible=bool(data_cfg.get("require_eligible", False)),
            )
        summary = evaluate_split_spec(
            model,
            dataset,
            threshold=threshold,
            device=device,
            amp=amp,
            batch_size=int(args.batch_size),
            out_dir=out_dir,
            split_name=name,
            tta_mode=str(args.tta),
            tta_weight=float(args.tta_weight),
        )
        summary["dataset"] = {
            "kind": kind,
            "spec": spec if isinstance(spec, (str, list)) else dict(spec),
            "num_records": len(dataset),
            "split_counts": dataset.split_counts(),
        }
        results[name] = summary
        print(
            f"EVAL {name} n={summary['n_rna']} exact_f1={summary['rna_macro_exact_f1']:.6f} "
            f"flexible_f1={summary['rna_macro_flexible_f1']:.6f} "
            f"family_macro_exact_f1={summary['family_macro_exact_f1']:.6f}",
            flush=True,
        )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_frozen_in_checkpoint": frozen_threshold,
        "device": str(device),
        "tta": {
            "mode": str(args.tta),
            "weight": float(args.tta_weight) if str(args.tta) != "none" else None,
            "views": 2 if (str(args.tta) == "rc" and float(args.tta_weight) > 0.0) else 1,
            "aggregation": "mean_probability" if str(args.tta) == "rc" else "none",
        },
        "splits": results,
        "elapsed_seconds": time.time() - started,
        "code_sha256": checkpoint.get("code_sha256"),
    }
    save_json(out_dir / "metrics.json", metrics)
    print(json.dumps({"out": str(out_dir), "splits": list(results)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
