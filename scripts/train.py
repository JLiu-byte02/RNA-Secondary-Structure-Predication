#!/usr/bin/env python
"""scripts/train.py -- config-driven end-to-end RiNALMo fine-tuning.

One run directory contains everything needed to reproduce a number:

    config.json          resolved config (merged with CLI overrides)
    code_sha256.json     sha256 of every source file that produced the run
    data_sha256.json     sha256 of every dataset file used
    train_log.txt        human-readable log
    train_log.jsonl      machine-readable per-epoch log
    metrics.json         selection history + final metrics on every eval split
    last.pt / best.pt    resume state / selected checkpoint
    predictions_<split>.jsonl   per-RNA predictions at the frozen threshold

Selection protocol (no test leakage):
  * every epoch, sweep ``selection.thresholds`` on the **select** split;
  * pick the threshold by ``selection.metric`` (ties -> smaller threshold);
  * the epoch is "best" if that metric improves (strictly);
  * the checkpoint + threshold are frozen before any eval split is touched.

Usage
-----
    python scripts/train.py --config my-run.yaml
    python scripts/train.py --config my-run.yaml --smoke --device cpu
    python scripts/train.py --config my-run.yaml --set model.backbone.mode=last_k
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as rna_data  # noqa: E402
from src.decode import decode_multi, select_threshold, threshold_grid  # noqa: E402
from src.loss import pair_bce_from_batch  # noqa: E402
from src.metrics import MetricAccumulator  # noqa: E402
from src.model import build_model, model_signature  # noqa: E402

RUN_FILES_SOURCE = ("src", "scripts/train.py", "scripts/evaluate.py", "scripts/eval_official.py")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"{path}: YAML root must be a mapping")
    return config


def deep_update(base: dict, override: Mapping[str, Any]) -> dict:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_override(config: dict, expression: str) -> None:
    """``a.b.c=value`` override; value parsed as YAML (so ``null``/``1e-4`` work)."""
    if "=" not in expression:
        raise ValueError(f"override {expression!r} is not key=value")
    key, raw = expression.split("=", 1)
    value = yaml.safe_load(raw)
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise TypeError(f"override {expression!r} crosses non-mapping key {part!r}")
    node[parts[-1]] = value


def resolve_device(spec: str) -> torch.device:
    spec = str(spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (use --device cpu for smoke tests)")
    return torch.device(spec)


def code_fingerprint(extra_files: Sequence[str | Path] = ()) -> dict:
    files: Dict[str, str] = {}
    for entry in RUN_FILES_SOURCE:
        path = ROOT / entry
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                files[str(child.relative_to(ROOT)).replace("\\", "/")] = sha256_file(child)
        elif path.is_file():
            files[entry] = sha256_file(path)
    for entry in extra_files:
        path = Path(entry)
        if path.is_file():
            files[str(path.resolve())] = sha256_file(path)
    combined = sha256_bytes(json.dumps(files, sort_keys=True).encode("utf-8"))
    return {"combined_sha256": combined, "files": files}


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataset(
    spec: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    channels: Sequence[str],
    thermo: rna_data.ThermoFeatureStore | None,
    alphabet,
    root: Path,
    limit: int | None = None,
) -> rna_data.RNAStructureDataset:
    data_cfg = cfg.get("data", {})
    return rna_data.build_dataset_for_spec(
        spec,
        root=root,
        channels=channels,
        thermo=thermo,
        alphabet=alphabet,
        limit=limit,
        min_sep=int(data_cfg.get("min_sep", rna_data.MIN_SEP)),
        max_length=data_cfg.get("max_length"),
        strict_sequence=bool(data_cfg.get("strict_sequence", True)),
        strict_pairs=bool(data_cfg.get("strict_pairs", True)),
        require_eligible=bool(data_cfg.get("require_eligible", True)),
    )


def build_datasets(
    cfg: Mapping[str, Any], model: torch.nn.Module, root: Path, smoke: bool = False
) -> tuple[Dict[str, rna_data.RNAStructureDataset], Dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    channels = list(model.data_channels())
    declared = list(data_cfg.get("channels") or [])
    if declared and sorted(declared) != sorted(channels):
        raise ValueError(
            f"data.channels {declared} != channels required by the head {channels}; "
            "the dataset must provide exactly what the model consumes"
        )
    thermo = None
    if any(channel in rna_data.THERMO_CHANNELS for channel in channels):
        directory = data_cfg.get("thermo_dir")
        if not directory:
            raise ValueError(
                f"head requires channels {channels} but data.thermo_dir is not configured"
            )
        thermo = rna_data.ThermoFeatureStore(
            (root / directory) if not Path(directory).is_absolute() else directory,
            channels=tuple(c for c in channels if c in rna_data.THERMO_CHANNELS),
            require_all=bool(data_cfg.get("require_thermo", False)),
        )
    alphabet = rna_data.get_alphabet(bool(data_cfg.get("use_rinalmo_alphabet", True)))

    specs = {
        "train": data_cfg["train"],
        "select": data_cfg["select"],
    }
    eval_specs = data_cfg.get("eval") or {}
    for name, spec in eval_specs.items():
        specs[f"eval::{name}"] = spec

    smoke_limit = int(data_cfg.get("smoke_max_records", 8)) if smoke else None
    datasets: Dict[str, rna_data.RNAStructureDataset] = {}
    stats: Dict[str, Any] = {}
    for name, spec in specs.items():
        limit = smoke_limit if (smoke and name in ("train", "select")) else None
        if smoke and name.startswith("eval::"):
            limit = smoke_limit
        if limit is None and data_cfg.get("max_records") and name in ("train", "select"):
            limit = int(data_cfg["max_records"])
        dataset = build_dataset(
            spec,
            cfg=cfg,
            channels=channels,
            thermo=thermo,
            alphabet=alphabet,
            root=root,
            limit=limit,
        )
        datasets[name] = dataset
        entry = dataset.stats()
        filter_report = getattr(dataset, "filter_report", None)
        if filter_report:
            entry["filter_report"] = filter_report
            print(
                f"WARNING {name}: dropped {filter_report['num_dropped']} records with "
                f"length > {filter_report['max_length']} (ids listed in config.json)",
                flush=True,
            )
        stats[name] = entry
        print(
            f"DATA {name} n={entry['num_records']} max_len={entry['max_length']} "
            f"ambiguous_bases={entry['ambiguous_bases']} "
            f"ineligible_ref_pairs={entry['ineligible_reference_pairs']}"
            f"/{entry['total_reference_pairs']}",
            flush=True,
        )
    return datasets, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_thresholds(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    thresholds: Sequence[float],
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> Dict[float, MetricAccumulator]:
    """One forward pass -> per-RNA decoded metrics for every threshold."""
    model.eval()
    accumulators = {float(threshold): MetricAccumulator() for threshold in thresholds}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch_to_device(batch, device)
        with autocast_context(device, amp):
            logits = model(
                batch["tokens"],
                lengths=batch["lengths"],
                padding_mask=batch["padding_mask"],
                channels=batch.get("channels"),
                pair_mask=batch["pair_mask"],
                channel_names=model.data_channels(),
            )
        logits = logits.float().cpu().numpy()
        lengths = batch["lengths"].tolist()
        for row, length in enumerate(lengths):
            sequence = batch["sequences"][row]
            scores = logits[row, :length, :length]
            decoded = decode_multi(scores, sequence, thresholds)
            reference = batch["pairs"][row]
            family = batch["families"][row]
            identifier = batch["ids"][row]
            for threshold, pairs in decoded.items():
                accumulators[float(threshold)].add(
                    identifier, family, reference, pairs, length=length
                )
        if (batch_index + 1) % 50 == 0:
            print(f"EVAL {batch_index + 1}/{len(loader)}", flush=True)
    return accumulators


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    threshold: float,
    device: torch.device,
    *,
    amp: bool,
    prediction_path: str | Path | None = None,
    split_name: str = "",
) -> dict:
    """Single-threshold evaluation; optionally dump per-RNA predictions."""
    model.eval()
    accumulator = MetricAccumulator()
    handle = None
    if prediction_path is not None:
        Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(prediction_path, "w", encoding="utf-8")
    try:
        for batch in loader:
            batch = batch_to_device(batch, device)
            with autocast_context(device, amp):
                logits = model(
                    batch["tokens"],
                    lengths=batch["lengths"],
                    padding_mask=batch["padding_mask"],
                    channels=batch.get("channels"),
                    pair_mask=batch["pair_mask"],
                    channel_names=model.data_channels(),
                )
            logits = logits.float().cpu().numpy()
            for row, length in enumerate(batch["lengths"].tolist()):
                sequence = batch["sequences"][row]
                scores = logits[row, :length, :length]
                pairs = decode_multi(scores, sequence, [threshold])[float(threshold)]
                row_metrics = accumulator.add(
                    batch["ids"][row],
                    batch["families"][row],
                    batch["pairs"][row],
                    pairs,
                    length=length,
                )
                if handle is not None:
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
                                "exact_f1": row_metrics["exact_f1"],
                                "flexible_f1": row_metrics["flexible_f1"],
                                "predicted_pairs": [list(pair) for pair in pairs],
                                "reference_pairs": [list(pair) for pair in batch["pairs"][row]],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if handle is not None:
            handle.close()
    return accumulator.summary()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_optimizer(model: torch.nn.Module, train_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    parameters = model.trainable_parameters()
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    if weight_decay <= 0.0:
        groups = [{"params": parameters, "weight_decay": 0.0}]
    else:
        no_decay = ("bias", "norm", "embedding")
        decay, no_decay_group = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(token in name for token in no_decay):
                no_decay_group.append(parameter)
            else:
                decay.append(parameter)
        groups = [{"params": decay, "weight_decay": weight_decay}]
        if no_decay_group:
            groups.append({"params": no_decay_group, "weight_decay": 0.0})
    betas = tuple(train_cfg.get("betas", [0.9, 0.999]))
    return optimizer_class(
        groups,
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=betas,
        fused=train_cfg.get("fused_optimizer"),
    )


def _schedule_for_epoch(schedule: Mapping[Any, Any], epoch: int) -> tuple[int, Any] | None:
    """Latest schedule entry whose (1-based) epoch <= current epoch."""
    best: tuple[int, Any] | None = None
    for key, value in schedule.items():
        start = int(key)
        if start <= epoch and (best is None or start > best[0]):
            best = (start, value)
    return best


def apply_unfreeze_schedule(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: Mapping[Any, Any] | None,
    epoch: int,
    *,
    base_lr: float,
    initial_denom_lr: float,
    weight_decay: float,
    entry_lr_mode: str = "base",
) -> dict | None:
    """Unfreeze encoder stages at epoch boundaries (official gradual unfreezing).

    Mirrors ``rinalmo/utils/finetune_callback.GradualUnfreezing``: newly
    unfrozen parameter groups are added to the optimizer with
    ``lr = base_lr / initial_denom_lr`` (Lightning's ``initial_denom_lr=10``),
    while already-trained groups keep their Adam state.
    """
    if not schedule:
        return None
    entry = _schedule_for_epoch(schedule, epoch)
    if entry is None:
        return None
    start, spec = entry
    if isinstance(spec, Mapping):
        mode = str(spec.get("mode", "frozen"))
        last_k = spec.get("last_k")
    else:
        mode, last_k = str(spec), None
    state_key = f"{start}:{mode}:{last_k}"
    if getattr(model, "_unfreeze_state_key", None) == state_key:
        return None
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "set_mode"):
        return None
    backbone.set_mode(mode, last_k=last_k)
    known = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    new_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in known
    ]
    if new_parameters:
        # Match Lightning's BaseFinetuning.unfreeze_and_add_param_group: when no
        # explicit lr is given the new group enters at *the current lr of the
        # first optimizer group* divided by initial_denom_lr -- not at the
        # configured base lr. With the official LinearLR schedule (lr has already
        # decayed to base/10 by epoch ~3) this makes later-unfrozen encoder
        # blocks train 10x more gently than dividing the config lr. Using the
        # config lr instead was measured to overfit the training families and
        # hurt the held-out family (5s test exact F1 0.827 vs the official 0.877).
        denom = max(float(initial_denom_lr), 1.0)
        if str(entry_lr_mode).lower() == "current":
            # Lightning 2.2's BaseFinetuning semantics (current group-0 lr / denom).
            entry_lr = float(optimizer.param_groups[0]["lr"]) / denom
        else:
            # Configured base lr / denom. Empirically better on the 5s fold:
            # official-head test exact F1 0.8272 (base, lr 1e-5) vs 0.8121
            # (current, lr 1e-6); the gentler entry under-adapts the encoder.
            entry_lr = float(base_lr) / denom
        optimizer.add_param_group(
            {
                "params": new_parameters,
                "lr": entry_lr,
                "weight_decay": float(weight_decay),
            }
        )
    model._unfreeze_state_key = state_key  # type: ignore[attr-defined]
    record = {
        "epoch": epoch,
        "schedule_epoch": start,
        "mode": mode,
        "last_k": last_k,
        "new_parameters": int(sum(parameter.numel() for parameter in new_parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    print(f"UNFREEZE {json.dumps(record)}", flush=True)
    return record


def loss_mode_from_config(train_cfg: Mapping[str, Any]) -> str:
    loss_cfg = train_cfg.get("loss", "eligible_candidates")
    if isinstance(loss_cfg, Mapping):
        return str(loss_cfg.get("type", "eligible_candidates"))
    return str(loss_cfg)


class TrainingLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def get_lr(self):
        scheduled_lrs = super().get_lr()
        # Groups added during unfreezing retain their entry learning rate.
        added_groups = self.optimizer.param_groups[len(scheduled_lrs) :]
        return scheduled_lrs + [group["lr"] for group in added_groups]


def make_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: Mapping[str, Any], total_steps: int
):
    warmup = int(train_cfg.get("warmup_steps", 0))
    schedule = str(train_cfg.get("scheduler", "linear")).lower()
    min_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    total_steps = max(int(total_steps), 1)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "cosine":
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
        if schedule in ("none", "constant"):
            return 1.0
        return max(min_ratio, 1.0 - (1.0 - min_ratio) * progress)

    return TrainingLRScheduler(optimizer, lr_lambda)


def selection_signature(datasets: Mapping[str, rna_data.RNAStructureDataset]) -> dict:
    signatures = {}
    for name, dataset in datasets.items():
        identifiers = [record["id"] for record in dataset.records]
        signatures[name] = {
            "count": len(identifiers),
            "sha256": sha256_bytes(json.dumps(sorted(identifiers)).encode("utf-8")),
            "lengths_sha256": sha256_bytes(
                json.dumps([[record["id"], record["length"]] for record in dataset.records]).encode("utf-8")
            ),
        }
    return signatures


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: Mapping[str, Any],
    epoch: int,
    global_step: int,
    threshold: float,
    best_score: float,
    best_epoch: int | None,
    stale: int,
    history: Sequence[Mapping[str, Any]],
    signature: str,
    data_sha256: Mapping[str, Any],
    code_sha256: Mapping[str, Any],
    include_optimizer: bool = True,
) -> dict:
    payload = {
        "format_version": 1,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "threshold": float(threshold),
        "best_score": float(best_score),
        "best_epoch": best_epoch,
        "stale": int(stale),
        "history": list(history),
        "model_signature": signature,
        "data_sha256": data_sha256,
        "code_sha256": code_sha256,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    return payload


def run_training(cfg: dict, args: argparse.Namespace) -> dict:
    started = time.time()
    run_name = str(args.run_name or cfg.get("run_name") or "run")
    output_root = Path(args.output_dir or cfg.get("output_dir") or "runs")
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 17))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    selection_cfg = cfg.get("selection", {})
    amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    if train_cfg.get("bf16") and device.type == "cpu":
        print("NOTE bf16 requested but device is CPU -> autocast disabled", flush=True)

    model = build_model(cfg["model"])
    model.to(device)
    signature = model_signature(model)
    counts = model.parameter_counts()
    datasets, data_stats = build_datasets(cfg, model, ROOT, smoke=bool(args.smoke))
    train_dataset = datasets["train"]
    select_dataset = datasets["select"]
    eval_datasets = {
        name.split("eval::", 1)[1]: dataset
        for name, dataset in datasets.items()
        if name.startswith("eval::")
    }
    if args.smoke:
        smoke_cap = int(data_cfg.get("smoke_max_length", 512))
        shortest = int(min(train_dataset.lengths))
        if shortest > smoke_cap:
            # never empty a smoke cohort just because lengths are large
            print(
                f"NOTE smoke_max_length={smoke_cap} < shortest training length {shortest}; "
                "raising the cap",
                flush=True,
            )
            smoke_cap = shortest
        train_dataset = train_dataset.filter_by_length(smoke_cap)

    code_sha = code_fingerprint([args.config])
    data_sha = {
        "train": rna_data.dataset_fingerprint(data_cfg["train"], ROOT),
        "select": rna_data.dataset_fingerprint(data_cfg["select"], ROOT),
        "eval": {
            name: rna_data.dataset_fingerprint(spec, ROOT)
            for name, spec in (data_cfg.get("eval") or {}).items()
        },
        "cohorts": selection_signature(datasets),
    }
    if data_cfg.get("thermo_dir"):
        thermo_records = {}
        for split_name, spec in {"train": data_cfg["train"], "select": data_cfg["select"], **(data_cfg.get("eval") or {})}.items():
            for path in rna_data.resolve_dataset_paths(spec, ROOT):
                thermo_records[str(path.relative_to(ROOT))] = sha256_file(path)
        data_sha["datasets"] = thermo_records

    batch_size = int(train_cfg.get("batch_size", 1))
    accumulation = max(int(train_cfg.get("grad_accum", 1)), 1)
    train_max_tokens = int(train_cfg.get("max_tokens_per_batch", 0) or 0) or None
    eval_max_tokens = int(train_cfg.get("eval_max_tokens_per_batch", 0) or 0) or None
    train_loader = rna_data.make_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        bucket_size=int(train_cfg.get("bucket_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=False,
        seed=seed,
        max_tokens=train_max_tokens,
    )
    select_loader = rna_data.make_dataloader(
        select_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
        shuffle=False,
        bucket_size=int(train_cfg.get("bucket_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        max_tokens=eval_max_tokens,
    )
    eval_loaders = {
        name: rna_data.make_dataloader(
            dataset,
            batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
            shuffle=False,
            bucket_size=int(train_cfg.get("bucket_size", 64)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            max_tokens=eval_max_tokens,
        )
        for name, dataset in eval_datasets.items()
    }

    thresholds = threshold_grid(
        selection_cfg.get("thresholds"),
        start=float(selection_cfg.get("threshold_start", 0.01)),
        stop=float(selection_cfg.get("threshold_stop", 0.30)),
        step=float(selection_cfg.get("threshold_step", 0.01)),
    )
    selection_metric = str(selection_cfg.get("metric", "rna_macro_official_flexible_f1"))
    loss_mode = loss_mode_from_config(train_cfg)
    print(f"CONFIG selection_metric={selection_metric} loss_mode={loss_mode}", flush=True)

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 10))
    if args.smoke:
        epochs = min(epochs, int(train_cfg.get("smoke_epochs", 2)))
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = int(
        train_cfg.get("scheduler_total_steps") or max(steps_per_epoch * epochs, 1)
    )
    optimizer = make_optimizer(model, train_cfg)
    scheduler = make_scheduler(optimizer, train_cfg, total_steps)
    unfreeze_schedule = train_cfg.get("unfreeze_schedule")
    unfreeze_denom_lr = float(train_cfg.get("unfreeze_initial_denom_lr", 10.0))
    base_lr = float(train_cfg.get("lr", 1e-4))

    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["data"] = dict(data_cfg)
    resolved_cfg["data"]["resolved_channels"] = list(model.data_channels())
    resolved_cfg["train"]["epochs"] = epochs
    resolved_cfg["runtime"] = {
        "seed": seed,
        "device": str(device),
        "amp_bf16": amp,
        "torch": torch.__version__,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "parameter_counts": counts,
        "model_signature": signature,
        "loss_mode": loss_mode,
        "selection_metric": selection_metric,
        "data_stats": data_stats,
    }

    history: List[dict] = []
    best_score = float("-inf")
    best_epoch = None
    best_threshold = thresholds[len(thresholds) // 2] if thresholds else 0.5
    stale = 0
    start_epoch = 1
    global_step = 0

    last_path = run_dir / "last.pt"
    resume_mode = str(args.resume).lower()
    if resume_mode != "never" and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_signature") != signature:
            raise RuntimeError(
                "Resume checkpoint architecture differs from this config "
                f"({checkpoint.get('model_signature')} vs {signature}); use a new run dir"
            )
        if checkpoint.get("data_sha256", {}).get("train") != data_sha.get("train"):
            raise RuntimeError("Resume checkpoint was trained on different training data")
        model.load_state_dict(checkpoint["model"])
        for stage in sorted(int(epoch) for epoch in (unfreeze_schedule or {})):
            if stage > int(checkpoint["epoch"]):
                break
            apply_unfreeze_schedule(
                model,
                optimizer,
                unfreeze_schedule,
                stage,
                base_lr=base_lr,
                initial_denom_lr=unfreeze_denom_lr,
                weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                entry_lr_mode=str(train_cfg.get("unfreeze_entry_lr", "base")),
            )
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_score = float(checkpoint.get("best_score", float("-inf")))
        best_epoch = checkpoint.get("best_epoch")
        best_threshold = float(checkpoint.get("threshold", best_threshold))
        stale = int(checkpoint.get("stale", 0))
        history = list(checkpoint.get("history", []))
        del checkpoint
        print(
            f"RESUME epoch={start_epoch} best_epoch={best_epoch} best={best_score:.6f}",
            flush=True,
        )

    # expose data stats / filtering so a reviewer can audit what was dropped
    resolved_cfg["data"]["stats"] = data_stats
    save_json(run_dir / "config.json", resolved_cfg)
    save_json(run_dir / "code_sha256.json", code_sha)
    save_json(run_dir / "data_sha256.json", data_sha)

    log_path = run_dir / "train_log.txt"
    jsonl_path = run_dir / "train_log.jsonl"
    log_handle = open(log_path, "a", encoding="utf-8")
    jsonl_handle = open(jsonl_path, "a", encoding="utf-8")
    log_handle.write(
        f"# run={run_name} seed={seed} device={device} epochs={epochs} "
        f"train={len(train_dataset)} select={len(select_dataset)} "
        f"params total={counts['total']} trainable={counts['trainable']}\n"
    )
    log_handle.flush()

    patience = int(train_cfg.get("patience", 0))
    log_every = int(train_cfg.get("log_every", 50))
    clip = float(train_cfg.get("clip_grad_norm", 1.0))
    pos_weight = train_cfg.get("pos_weight")
    save_every_epoch = bool(train_cfg.get("save_every_epoch", False))
    completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, epochs + 1):
            if patience and stale >= patience:
                print(f"EARLY_STOP stale={stale} patience={patience}", flush=True)
                break
            unfreeze_record = apply_unfreeze_schedule(
                model,
                optimizer,
                unfreeze_schedule,
                epoch,
                base_lr=base_lr,
                initial_denom_lr=unfreeze_denom_lr,
                weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                entry_lr_mode=str(train_cfg.get("unfreeze_entry_lr", "base")),
            )
            model.train()
            train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[attr-defined]
            order_started = time.time()
            loss_sum, counted = 0.0, 0
            micro_in_window = 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader):
                batch = batch_to_device(batch, device)
                with autocast_context(device, amp):
                    logits = model(
                        batch["tokens"],
                        lengths=batch["lengths"],
                        padding_mask=batch["padding_mask"],
                        channels=batch.get("channels"),
                        pair_mask=batch["pair_mask"],
                        channel_names=model.data_channels(),
                    )
                    loss = pair_bce_from_batch(
                        logits, batch, pos_weight=pos_weight, mode=loss_mode
                    )
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch} step {step}"
                    )
                (loss / accumulation).backward()
                micro_in_window += 1
                loss_sum += float(loss.detach())
                counted += 1
                if micro_in_window == accumulation:
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.trainable_parameters(), clip, error_if_nonfinite=True
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    micro_in_window = 0
                    global_step += 1
                if log_every and (step + 1) % log_every == 0:
                    mean_loss = loss_sum / max(counted, 1)
                    print(
                        f"TRAIN epoch={epoch} step={step + 1}/{len(train_loader)} "
                        f"loss={mean_loss:.6f} lr={optimizer.param_groups[0]['lr']:.3e}",
                        flush=True,
                    )
            if micro_in_window:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulation / micro_in_window)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.trainable_parameters(), clip, error_if_nonfinite=True
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if counted == 0:
                raise RuntimeError("no trainable sample in this epoch (all empty candidate sets?)")
            train_seconds = time.time() - order_started

            eval_started = time.time()
            selection_accumulators = evaluate_thresholds(
                model, select_loader, thresholds, device, amp=amp
            )
            selection_metrics = {
                threshold: accumulator.summary()
                for threshold, accumulator in selection_accumulators.items()
            }
            epoch_threshold, epoch_score = select_threshold(selection_metrics, selection_metric)
            improved = epoch_score > best_score
            if improved:
                best_score, best_epoch, best_threshold, stale = epoch_score, epoch, epoch_threshold, 0
            else:
                stale += 1
            completed_epoch = epoch
            eval_seconds = time.time() - eval_started

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "mean_train_loss": loss_sum / counted,
                "n_micro_batches": counted,
                "lr": optimizer.param_groups[0]["lr"],
                "select_threshold": epoch_threshold,
                "select_score": epoch_score,
                "select_metric": selection_metric,
                "select_metrics": selection_metrics[epoch_threshold],
                "official_flexible_f1": selection_metrics[epoch_threshold].get(
                    "rna_macro_official_flexible_f1"
                ),
                "exact_f1": selection_metrics[epoch_threshold].get("rna_macro_exact_f1"),
                "unfreeze": unfreeze_record,
                "is_best": improved,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale": stale,
                "train_seconds": train_seconds,
                "select_seconds": eval_seconds,
            }
            history.append(record)
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=resolved_cfg,
                epoch=epoch,
                global_step=global_step,
                threshold=epoch_threshold,
                best_score=best_score,
                best_epoch=best_epoch,
                stale=stale,
                history=history,
                signature=signature,
                data_sha256=data_sha,
                code_sha256=code_sha,
            )
            save_checkpoint(last_path, payload)
            if improved:
                payload_best = dict(payload)
                payload_best.pop("optimizer", None)
                payload_best.pop("scheduler", None)
                save_checkpoint(run_dir / "best.pt", payload_best)
            if save_every_epoch:
                # weights-only snapshot for weight soup (no optimizer state)
                payload_epoch = {
                    key: value
                    for key, value in payload.items()
                    if key not in ("optimizer", "scheduler")
                }
                save_checkpoint(run_dir / f"epoch{epoch}.pt", payload_epoch)
            line = (
                f"epoch={epoch} loss={record['mean_train_loss']:.6f} "
                f"select_{selection_metric}={epoch_score:.6f} threshold={epoch_threshold:.2f} "
                f"best={best_score:.6f}@{best_epoch} stale={stale} "
                f"train_s={train_seconds:.1f} eval_s={eval_seconds:.1f}"
            )
            print(f"EPOCH {line}", flush=True)
            log_handle.write(line + "\n")
            log_handle.flush()
            jsonl_handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            jsonl_handle.flush()
            metrics_snapshot = {
                "run_name": run_name,
                "status": "running",
                "epochs_completed": completed_epoch,
                "best_epoch": best_epoch,
                "threshold": best_threshold,
                "selection_metric": selection_metric,
                "best_score": best_score,
                "selection_history": history,
            }
            save_json(run_dir / "metrics.json", metrics_snapshot)
    except Exception:
        save_json(
            run_dir / "status.json",
            {"status": "failed", "epochs_completed": completed_epoch, "run_name": run_name},
        )
        raise
    finally:
        log_handle.close()
        jsonl_handle.close()

    if best_epoch is None:
        raise RuntimeError("training produced no best checkpoint")

    best_path = run_dir / "best.pt"
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    del best_checkpoint["model"]
    threshold = float(best_checkpoint["threshold"])
    print(f"FROZEN best_epoch={best_checkpoint['epoch']} threshold={threshold:.2f}", flush=True)
    # Freeze the selection decision *before* any eval split is touched.
    selection_lock = {
        "selected_epoch": int(best_checkpoint["epoch"]),
        "threshold": threshold,
        "selection_metric": selection_metric,
        "best_score": float(best_checkpoint.get("best_score", best_score)),
        "selected_on": "select split only",
        "test_splits_touched_before_lock": False,
        "threshold_grid": thresholds,
        "history": history,
    }
    save_json(run_dir / "selection_lock.json", selection_lock)

    write_predictions = bool(cfg.get("decode", {}).get("write_predictions", True))
    split_metrics = {}
    for name, loader in eval_loaders.items():
        prediction_path = run_dir / f"predictions_{name}.jsonl" if write_predictions else None
        summary = evaluate_split(
            model,
            loader,
            threshold,
            device,
            amp=amp,
            prediction_path=prediction_path,
            split_name=name,
        )
        split_metrics[name] = summary
        print(
            f"EVAL split={name} exact_f1={summary['rna_macro_exact_f1']:.6f} "
            f"flexible_f1={summary['rna_macro_flexible_f1']:.6f} "
            f"official_flexible_f1={summary['rna_macro_official_flexible_f1']:.6f} "
            f"family_macro_exact_f1={summary['family_macro_exact_f1']:.6f}",
            flush=True,
        )

    elapsed = time.time() - started
    metrics = {
        "run_name": run_name,
        "status": "completed",
        "seed": seed,
        "device": str(device),
        "epochs_completed": completed_epoch,
        "best_epoch": best_checkpoint["epoch"],
        "threshold": threshold,
        "selection_metric": selection_metric,
        "best_score": best_score,
        "split_metrics": split_metrics,
        "selection_history": history,
        "parameter_counts": counts,
        "model_signature": signature,
        "data_sha256": data_sha,
        "code_sha256": code_sha["combined_sha256"],
        "runtime": resolved_cfg["runtime"],
        "data_stats": data_stats,
        "loss_mode": loss_mode,
        "selection_lock": selection_lock,
        "elapsed_seconds": elapsed,
        "config_path": str(Path(args.config)),
    }
    save_json(run_dir / "metrics.json", metrics)
    save_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "run_name": run_name,
            "best_epoch": best_checkpoint["epoch"],
            "threshold": threshold,
            "elapsed_seconds": elapsed,
            "split_metrics": {
                name: {
                    "rna_macro_exact_f1": value["rna_macro_exact_f1"],
                    "rna_macro_flexible_f1": value["rna_macro_flexible_f1"],
                    "family_macro_exact_f1": value["family_macro_exact_f1"],
                    "n_rna": value["n_rna"],
                }
                for name, value in split_metrics.items()
            },
        },
    )
    print(f"RUN_COMPLETED {json.dumps({'run_dir': str(run_dir), 'threshold': threshold, 'best_epoch': best_checkpoint['epoch']})}", flush=True)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--run-name", default=None, help="override run_name")
    parser.add_argument("--output-dir", default=None, help="override output_dir")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--resume",
        default="auto",
        choices=["auto", "never"],
        help="auto resumes last.pt when present; never starts fresh",
    )
    parser.add_argument("--smoke", action="store_true", help="2 epochs, tiny cohorts, CPU-friendly")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="a.b.c=value override")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    for override in args.overrides:
        apply_override(config, override)
    if args.smoke:
        config.setdefault("decode", {})["write_predictions"] = True
    metrics = run_training(config, args)
    print(json.dumps({"run_name": metrics["run_name"], "threshold": metrics["threshold"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
