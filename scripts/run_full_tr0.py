from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import build_candidates, load_embeddings, load_manifest, load_split, set_seed
from src.decoder import candidate_pairs, decode_multi
from src.losses import pair_bce_loss
from src.metrics import family_macro, pair_prf, rna_macro
from src.pair_head import PairHead
from src.utils import pair_set_0b, pairs_from_str

from scripts.run_thermo_fast import (MODES, ThermoFusion, digest, input_hash, state_hash,
    save_json, save_checkpoint, build_records, model_inputs, evaluate)
from scripts.full_protocol import validate_config

def train_locked(args):
    root = Path(args.root).resolve()
    if root != ROOT.resolve():
        raise RuntimeError("--root must match this source checkout")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    seed = args.seed if args.seed is not None else config["seed"]
    configuration_hash = digest(config)
    output_root = root / "outputs/full_tr0"
    run_path = output_root / ("smoke" if args.smoke else "") / args.mode / f"seed{seed}"
    run_path.mkdir(parents=True, exist_ok=True)
    if (run_path / "summary.json").exists():
        summary = json.loads((run_path / "summary.json").read_text())
        if summary.get("status") == "completed" and summary.get("config_hash") == configuration_hash:
            if "input_provenance_hash" in summary:
                current_audit = json.loads((output_root / "split_audit.json").read_text())
                current_cache = json.loads((output_root / "thermo_cache_manifest.json").read_text())
                if summary["input_provenance_hash"] != input_hash(root, current_audit, current_cache):
                    raise RuntimeError("Completed inputs differ from this request")
                previous_source = json.loads((run_path / "input_provenance.json").read_text())["source_sha256"]
                current_source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                if previous_source != current_source:
                    compatibility_path = output_root / "provenance/source_compatibility.json"
                    compatibility = json.loads(compatibility_path.read_text()) if compatibility_path.exists() else {}
                    if compatibility.get("current_source_sha256") != current_source or previous_source not in compatibility.get("compatible_source_sha256", []):
                        raise RuntimeError("Completed run uses an incompatible source snapshot; preserve it and reproduce in an isolated project subdirectory")
            expected_predictions = {"VL0_predictions.jsonl": summary["dev_count"]}
            for filename, expected_count in expected_predictions.items():
                prediction_file = run_path / filename
                if expected_count and (not prediction_file.exists() or sum(1 for line in prediction_file.open() if line.strip()) != expected_count):
                    raise RuntimeError("Completed predictions missing or incomplete: " + str(prediction_file))
            if not (run_path / "best.pt").exists():
                raise RuntimeError("Completed checkpoint is missing")
            print("ALREADY_COMPLETED", json.dumps(summary), flush=True)
            return
        raise RuntimeError("Existing results have different configuration: " + str(run_path))
    set_seed(seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This audited protocol requires CUDA BF16 support; adapt uniformly before any formal run")
    device = torch.device("cuda")
    audit = json.loads((output_root / "split_audit.json").read_text())
    if not audit.get("passed"):
        raise RuntimeError("Split audit has not passed")
    for relative, key in (("data/processed/manifest.tsv", "manifest_sha256"), ("data/processed/split_ids.json", "split_ids_sha256")):
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != audit[key]:
            raise RuntimeError("Input changed after the split audit: " + relative)
    cache_manifest = json.loads((output_root / "thermo_cache_manifest.json").read_text())
    input_provenance_hash = input_hash(root, audit, cache_manifest)
    features = cache_manifest["samples"]
    manifest = load_manifest(root)
    split = load_split(root)
    train_ids = [identifier for identifier in split["TR0"] if features.get(identifier, {}).get("status") == "success"]
    dev_ids = [identifier for identifier in split["VL0"] if features.get(identifier, {}).get("status") == "success"]
    if train_ids != split['TR0'] or dev_ids != split['VL0']:
        raise RuntimeError('Full stage requires every fixed training and validation sample')
    vl_ids = []
    if args.smoke:
        train_ids = train_ids[:32]
        dev_ids = dev_ids[:16]
        vl_ids = []
    if not train_ids or not dev_ids:
        raise RuntimeError("Empty required shared successful cohort")
    identifiers = train_ids + dev_ids + vl_ids
    records = build_records(root, identifiers, manifest, features)
    embeddings = {}
    for name, requested in (("TR0", train_ids), ("VL0", dev_ids)):
        if requested:
            embedding_path = root / "embeddings/full" / f"{name}.pt"
            from scripts.cache_thermo import file_hash
            if file_hash(embedding_path) != audit['embedding_sha256'][name]:
                raise RuntimeError('Full-stage embedding checksum changed: '+name)
            cached = torch.load(embedding_path, map_location="cpu", weights_only=False)
            for identifier in requested:
                value = cached[identifier]
                if value.shape != (len(records[identifier]["sequence"]), 1280):
                    raise ValueError("Embedding shape mismatch: " + identifier)
                if value.requires_grad:
                    raise ValueError("Cached encoder embedding must be frozen")
                embeddings[identifier] = value
            del cached
    model = ThermoFusion(args.mode).to(device)
    initialization_hash = state_hash(model.state_dict())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    probe_record = records[train_ids[0]]
    with torch.no_grad():
        initial_corrected, initial_logits, residual = model(*model_inputs(probe_record, embeddings[train_ids[0]], device), return_initial=True)
        if residual.abs().max().item() != 0 or not torch.equal(initial_corrected[probe_record["valid"].to(device)], initial_logits[probe_record["valid"].to(device)]):
            raise AssertionError("Zero initialization does not reproduce B0")
        if not torch.equal(initial_corrected, initial_corrected.T):
            raise AssertionError("Asymmetric logits")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["initial_lr"], weight_decay=config["weight_decay"])
    status = dict(mode=args.mode, seed=seed, smoke=args.smoke, status="running", pid=os.getpid(),
                  config_hash=configuration_hash, initialization_hash=initialization_hash,
                  parameter_count=parameter_count, train_count=len(train_ids), dev_count=len(dev_ids),
                  VL0_count=len(dev_ids), train_ids_hash=digest(train_ids), dev_ids_hash=digest(dev_ids),
                  encoder_frozen=True, loaded_supervised_checkpoint=False)
    save_json(run_path / "status.json", status)
    save_json(run_path / "config.json", config)
    save_json(run_path / "resolved_config.json", dict(config, seed=seed, mode=args.mode, smoke=args.smoke))
    save_json(run_path / "cohort.json", dict(TR0=train_ids, VL0=dev_ids))
    source_bytes = Path(__file__).read_bytes()
    source_fingerprint = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for sub in ("src", "scripts") for p in (root/sub).glob("*.py")}
    code_identity = digest(source_fingerprint)
    code_record = run_path / "code_identity.json"
    if code_record.exists() and json.loads(code_record.read_text())["sha256"] != code_identity:
        raise RuntimeError("Refuse resume with changed source; preserve run and use a new run directory")
    save_json(code_record, dict(sha256=code_identity, files=source_fingerprint))
    if not (run_path / "run_source.py").exists():
        (run_path / "run_source.py").write_bytes(source_bytes)
        save_json(run_path / "input_provenance.json", dict(input_provenance_hash=input_provenance_hash,
                  source_sha256=hashlib.sha256(source_bytes).hexdigest(), audit_path="outputs/full_tr0/split_audit.json",
                  cache_manifest_path="outputs/full_tr0/thermo_cache_manifest.json"))
    elif json.loads((run_path / "input_provenance.json").read_text())["input_provenance_hash"] != input_provenance_hash:
        raise RuntimeError("Input provenance changed since this run started")
    print("RUN_START", json.dumps(status), flush=True)
    max_epochs = 1 if args.smoke else config["max_epochs"]
    accumulation = config["gradient_accumulation"]
    thresholds = config["thresholds"]
    best_score, best_epoch, best_threshold, stale, start_epoch = -1.0, None, None, 0, 1
    training_seconds, evaluation_seconds = 0.0, 0.0
    last_path = run_path / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["initialization_hash"] != initialization_hash or checkpoint["config_hash"] != configuration_hash or checkpoint["train_ids_hash"] != digest(train_ids):
            raise RuntimeError("Resume provenance mismatch")
        if checkpoint.get("dev_ids_hash", digest(dev_ids)) != digest(dev_ids) or checkpoint.get("VL0_ids_hash", digest(dev_ids)) != digest(dev_ids):
            raise RuntimeError("Resume selection/monitoring cohort mismatch")
        if checkpoint.get("input_provenance_hash", input_provenance_hash) != input_provenance_hash:
            raise RuntimeError("Resume input provenance mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_score, best_epoch, best_threshold = checkpoint["best_score"], checkpoint["best_epoch"], checkpoint["best_threshold"]
        stale, start_epoch = checkpoint["stale"], checkpoint["epoch"] + 1
        training_seconds, evaluation_seconds = checkpoint["training_seconds"], checkpoint["evaluation_seconds"]
        print("RESUME", start_epoch, flush=True)
    gradients_verified = checkpoint.get("gradients_verified", False) if last_path.exists() else False
    completed_epoch = start_epoch - 1
    with (run_path / "train_log.jsonl").open("a", encoding="utf8") as log:
        for epoch in range(start_epoch, max_epochs + 1):
            if stale >= config["patience"]:
                break
            model.train()
            order = list(train_ids)
            random.Random(seed * 1000 + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            loss_sum, counted, micro = 0.0, 0, 0
            started = time.time()
            for step, identifier in enumerate(order):
                record = records[identifier]
                if not record["cand_i"].numel():
                    continue
                logits = model(*model_inputs(record, embeddings[identifier], device))
                loss = pair_bce_loss(logits, record["cand_i"].to(device), record["cand_j"].to(device), record["labels"].to(device))
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(identifier)
                (loss / accumulation).backward()
                if not gradients_verified:
                    head_grads = [parameter.grad for parameter in model.head.parameters() if parameter.grad is not None]
                    cnn_grads = [parameter.grad for parameter in model.correction.parameters() if parameter.grad is not None]
                    if not head_grads or not cnn_grads or not all(torch.isfinite(value).all() for value in head_grads + cnn_grads):
                        raise AssertionError("Head/CNN gradients missing or nonfinite")
                    if not any(value.abs().max() > 0 for value in head_grads) or not any(value.abs().max() > 0 for value in cnn_grads):
                        raise AssertionError("Head/CNN gradients are zero")
                    gradients_verified = True
                micro += 1
                counted += 1
                loss_sum += loss.item()
                if micro == accumulation:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    micro = 0
                if (step + 1) % 100 == 0:
                    status.update(epoch=epoch, step=step + 1, training_mean_loss=loss_sum / counted,
                                  epoch_elapsed_seconds=time.time() - started)
                    save_json(run_path / "status.json", status)
                    print("TRAIN", args.mode, seed, epoch, step + 1, len(order), "loss", round(loss_sum / counted, 6), "seconds", round(time.time() - started, 1), flush=True)
            if micro:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulation / micro)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not counted:
                raise RuntimeError("No valid training samples")
            epoch_training_seconds = time.time() - started
            training_seconds += epoch_training_seconds
            started = time.time()
            evaluation = evaluate(model, dev_ids, embeddings, records, thresholds, device)
            epoch_evaluation_seconds = time.time() - started
            evaluation_seconds += epoch_evaluation_seconds
            threshold = max(thresholds, key=lambda value: evaluation[value]["rna_macro_exact_f1"])
            score = evaluation[threshold]["rna_macro_exact_f1"]
            improved = score > best_score
            if improved:
                best_score, best_epoch, best_threshold, stale = score, epoch, threshold, 0
            else:
                stale += 1
            completed_epoch = epoch
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), mode=args.mode,
                              seed=seed, epoch=epoch, threshold=threshold, best_score=best_score,
                              best_epoch=best_epoch, best_threshold=best_threshold, stale=stale,
                              initialization_hash=initialization_hash, config_hash=configuration_hash,
                              train_ids_hash=digest(train_ids), training_seconds=training_seconds,
                              evaluation_seconds=evaluation_seconds, config=config,
                              gradients_verified=gradients_verified, dev_ids_hash=digest(dev_ids),
                              VL0_ids_hash=digest(dev_ids), input_provenance_hash=input_provenance_hash)
            save_checkpoint(run_path / f"epoch{epoch}.pt", checkpoint)
            save_checkpoint(last_path, checkpoint)
            if improved:
                save_checkpoint(run_path / "best.pt", checkpoint)
            record = dict(epoch=epoch, mean_pair_loss=loss_sum / counted, n_steps=counted,
                          threshold=threshold, is_best=improved, stale=stale,
                          training_seconds=epoch_training_seconds, evaluation_seconds=epoch_evaluation_seconds,
                          metrics=evaluation[threshold], threshold_metrics=evaluation)
            log.write(json.dumps(record) + "\n")
            log.flush()
            status.update(epoch=epoch, step=len(order), best_epoch=best_epoch, best_threshold=best_threshold,
                          best_VL0_rna_macro_exact_f1=best_score, elapsed_training_seconds=training_seconds,
                          elapsed_evaluation_seconds=evaluation_seconds)
            save_json(run_path / "status.json", status)
            print("EPOCH_COMPLETE", json.dumps(record), flush=True)
    if best_epoch is None:
        raise RuntimeError("No trained checkpoint available")
    best = torch.load(run_path / f"epoch{best_epoch}.pt", map_location=device, weights_only=False)
    if best["epoch"] != best_epoch or best["threshold"] != best_threshold:
        raise RuntimeError("Selected checkpoint metadata mismatch")
    save_checkpoint(run_path / "best.pt", best)
    model.load_state_dict(best["model"])
    started = time.time()
    validation_curve = evaluate(model, dev_ids, embeddings, records, thresholds, device)
    save_json(run_path / "validation_curve.json", {str(t):m["rna_macro_exact_f1"] for t,m in validation_curve.items()})
    dev_metrics = evaluate(model, dev_ids, embeddings, records, [best_threshold], device, run_path / "VL0_predictions.jsonl")[best_threshold]
    
    evaluation_seconds += time.time() - started
    summary = dict(status="completed", mode=args.mode, seed=seed, smoke=args.smoke,
                   best_epoch=best_epoch, threshold=best_threshold, epochs_completed=completed_epoch,
                   **dev_metrics, VL0_exact_f1=dev_metrics["rna_macro_exact_f1"],
                   VL0_original_exact_f1=dev_metrics["original_rna_macro_exact_f1"],
                   parameter_count=parameter_count, initialization_hash=initialization_hash,
                   config_hash=configuration_hash, training_seconds=training_seconds,
                   evaluation_seconds=evaluation_seconds, gradients_verified=gradients_verified,
                   train_count=len(train_ids), dev_count=len(dev_ids), VL0_count=len(dev_ids),
                   train_ids_hash=digest(train_ids), dev_ids_hash=digest(dev_ids),
                   VL0_ids_hash=digest(dev_ids),
                   input_provenance_hash=input_provenance_hash,
                   supervised_initialization="none_identical_random", encoder_frozen=True,
                   label_policy=config["label_policy"], test_data_evaluated=False)
    save_json(run_path / "summary.json", summary)
    save_json(run_path / "status.json", summary)
    print("RUN_COMPLETED", json.dumps(summary), flush=True)


def train(args):
    root = Path(args.root).resolve()
    if root != ROOT.resolve():
        raise RuntimeError("--root must match this source checkout")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    seed = args.seed if args.seed is not None else config["seed"]
    run_path = root / "outputs/full_tr0" / ("smoke" if args.smoke else "") / args.mode / f"seed{seed}"
    run_path.mkdir(parents=True, exist_ok=True)
    with (run_path / ".run.lock").open("a") as lock:
        print("WAIT_FOR_RUN_LOCK", args.mode, seed, flush=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            train_locked(args)
        except Exception:
            error = traceback.format_exc()
            save_json(run_path / "status.json", dict(status="failed", mode=args.mode, seed=seed, error=error))
            with (run_path / "errors.log").open("a") as handle:
                handle.write(error + "\n")
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--config', default='configs/full_tr0.yaml')
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, choices=[17,23,41], required=True)
    parser.add_argument('--smoke', action='store_true')
    train(parser.parse_args())
