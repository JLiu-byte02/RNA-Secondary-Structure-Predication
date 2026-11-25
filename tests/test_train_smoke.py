"""End-to-end CPU smoke: train.py -> checkpoints/metrics -> evaluate.py.

Eight synthetic RNAs, two epochs, stub 1280-d backbone, CPU only.  Asserts the
run directory contract, that the training loss actually decreases, and that the
frozen checkpoint can be re-scored by ``scripts/evaluate.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
import torch

from tests.helpers import synthetic_records, write_jsonl_gz

ROOT = Path(__file__).resolve().parents[1]


def _write_smoke_config(tmp_path: Path) -> Path:
    train = write_jsonl_gz(
        tmp_path / "data" / "tr0.jsonl.gz",
        synthetic_records(8, seed=11, split="TR0", families=("famA", "famB")),
    )
    select = write_jsonl_gz(
        tmp_path / "data" / "vl0.jsonl.gz",
        synthetic_records(6, seed=12, split="VL0", families=("famA", "famB")),
    )
    test = write_jsonl_gz(
        tmp_path / "data" / "ts0.jsonl.gz",
        synthetic_records(5, seed=13, split="TS0", families=("famA", "famB")),
    )
    config = {
        "run_name": "smoke",
        "seed": 3,
        "device": "cpu",
        "data": {
            "train": {"glob": str(train)},
            "select": {"glob": str(select)},
            "eval": {"TS0": {"glob": str(test)}},
            "channels": [],
            "min_sep": 4,
            "strict_sequence": False,
            "strict_pairs": False,
            "require_eligible": False,
            "use_rinalmo_alphabet": False,
            "num_workers": 0,
        },
        "model": {
            "backbone": {"type": "stub", "embed_dim": 1280, "num_layers": 2, "num_heads": 20},
            "head": {"type": "outer_product", "kwargs": {"conv_dim": 64, "num_blocks": 2}},
            "channels": [],
        },
        "train": {
            "epochs": 2,
            "batch_size": 2,
            "eval_batch_size": 2,
            "bucket_size": 8,
            "grad_accum": 1,
            "optimizer": "adamw",
            "loss": "eligible_candidates",
            "lr": 0.001,
            "weight_decay": 0.01,
            "scheduler": "none",
            "warmup_steps": 0,
            "clip_grad_norm": 1.0,
            "bf16": False,
            "patience": 0,
            "log_every": 1,
            "num_workers": 0,
            "save_every_epoch": True,
            "smoke_epochs": 2,
            "smoke_max_records": 8,
            "smoke_max_length": 96,
        },
        "selection": {
            "metric": "rna_macro_official_flexible_f1",
            "thresholds": [0.05, 0.1, 0.2, 0.3, 0.5],
        },
        "decode": {"min_sep": 4, "write_predictions": True},
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run(args, timeout: int = 900) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_train_then_evaluate_end_to_end(tmp_path):
    config_path = _write_smoke_config(tmp_path)
    output_dir = tmp_path / "runs"
    result = _run(
        [
            "scripts/train.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "smoke",
        ]
    )
    assert result.returncode == 0, (
        f"train failed\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
    )
    run_dir = output_dir / "smoke"
    for name in (
        "config.json",
        "code_sha256.json",
        "data_sha256.json",
        "metrics.json",
        "train_log.txt",
        "train_log.jsonl",
        "last.pt",
        "best.pt",
        "selection_lock.json",
    ):
        assert (run_dir / name).exists(), f"missing {name}"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "completed"
    assert metrics["threshold"] > 0
    assert metrics["loss_mode"] == "eligible_candidates"
    assert metrics["selection_metric"] == "rna_macro_official_flexible_f1"
    assert "TS0" in metrics["split_metrics"]
    assert metrics["split_metrics"]["TS0"]["n_rna"] == 5
    assert metrics["data_stats"]["train"]["num_records"] == 8
    assert metrics["data_stats"]["train"]["ambiguous_bases"] == 0

    lock = json.loads((run_dir / "selection_lock.json").read_text(encoding="utf-8"))
    assert lock["test_splits_touched_before_lock"] is False
    assert lock["threshold"] == metrics["threshold"]

    epoch_records = [
        json.loads(line)
        for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(epoch_records) == 2
    assert epoch_records[-1]["mean_train_loss"] < epoch_records[0]["mean_train_loss"], (
        f"loss did not decrease: {[record['mean_train_loss'] for record in epoch_records]}"
    )

    predictions = (run_dir / "predictions_TS0.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(predictions) == 5
    row = json.loads(predictions[0])
    assert {"id", "predicted_pairs", "reference_pairs", "exact_f1", "flexible_f1"} <= set(row)

    # per-epoch weights-only snapshots for checkpoint soup
    for epoch in (1, 2):
        epoch_path = run_dir / f"epoch{epoch}.pt"
        assert epoch_path.exists(), f"missing epoch{epoch}.pt"
        epoch_payload = torch.load(epoch_path, map_location="cpu", weights_only=False)
        assert "optimizer" not in epoch_payload and "scheduler" not in epoch_payload
        assert epoch_payload["epoch"] == epoch
        assert epoch_payload["model"]

    # frozen checkpoint -> standalone evaluation, threshold must come from the checkpoint
    evaluation_dir = tmp_path / "eval"
    result = _run(
        [
            "scripts/evaluate.py",
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--split-file",
            f"TS0={tmp_path / 'data' / 'ts0.jsonl.gz'}",
            "--out",
            str(evaluation_dir),
            "--device",
            "cpu",
        ]
    )
    assert result.returncode == 0, (
        f"evaluate failed\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
    )
    evaluation = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    assert evaluation["threshold_source"] == "checkpoint"
    assert evaluation["threshold"] == metrics["threshold"]
    assert evaluation["splits"]["TS0"]["n_rna"] == 5
    assert len((evaluation_dir / "predictions_TS0.jsonl").read_text().splitlines()) == 5


def test_official_recipe_unfreeze_schedule_runs_on_cpu(tmp_path):
    """The gradual-unfreezing schedule must be exercised end-to-end on the stub."""
    config = yaml.safe_load(_write_smoke_config(tmp_path).read_text(encoding="utf-8"))
    config["model"]["backbone"].update({"type": "stub", "embed_dim": 128, "num_layers": 6, "num_heads": 4})
    config["model"]["head"]["kwargs"] = {"conv_dim": 16, "num_blocks": 1}
    config["train"].update(
        {
            "epochs": 3,
            "loss": "full_upper_triangle",
            "optimizer": "adam",
            "weight_decay": 0.0,
            "unfreeze_schedule": {
                1: {"mode": "frozen"},
                2: {"mode": "last_k", "last_k": 2},
                3: {"mode": "last_k", "last_k": 4},
            },
        }
    )
    path = tmp_path / "schedule.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = _run(
        [
            "scripts/train.py",
            "--config",
            str(path),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "runs_schedule"),
            "--run-name",
            "schedule",
        ]
    )
    assert result.returncode == 0, (
        f"schedule train failed\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "runs_schedule" / "schedule" / "train_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    unfreeze_records = [record["unfreeze"] for record in records if record.get("unfreeze")]
    assert [entry["last_k"] for entry in unfreeze_records] == [None, 2, 4]
    assert unfreeze_records[0]["mode"] == "frozen"
    trainable = [entry["trainable_parameters"] for entry in unfreeze_records]
    assert trainable[0] < trainable[1] < trainable[2], trainable
