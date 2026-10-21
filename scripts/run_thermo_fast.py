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

MODES = ("E0_no_thermo", "E1_bpp", "E2_bpp_stack")
PRODUCT_MODE = "E3_bpp_product"

def marginal_product(bpp):
    inner = torch.zeros_like(bpp)
    inner[:-1, 1:] = bpp[1:, :-1]
    upper = torch.triu(bpp * inner, diagonal=1)
    return upper + upper.T



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def input_hash(root, audit, cache):
    feature_identity = dict(version=cache["viennarna_version"], params=cache["params"],
                            cutoff=cache["stack_cutoff"], features=cache["features"],
                            samples={identifier: {key: value.get(key) for key in ("status", "sequence_hash", "cache_path", "cache_sha256", "split")}
                                     for identifier, value in cache["samples"].items()})
    return digest(dict(manifest=hashlib.sha256((root / "data/processed/manifest.tsv").read_bytes()).hexdigest(),
                       split=hashlib.sha256((root / "data/processed/split_ids.json").read_bytes()).hexdigest(),
                       embeddings=audit["embedding_sha256"], features=feature_identity))


def state_hash(state):
    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        hasher.update(name.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class CorrectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, channels):
        return self.layers(channels.unsqueeze(0))[0, 0]


class ThermoFusion(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.head = PairHead()
        self.correction = CorrectionCNN()

    def forward(self, embedding, physical, distance, valid_mask, return_initial=False):
        initial = self.head(embedding).float()
        if not torch.isfinite(initial).all():
            raise FloatingPointError("Nonfinite unmasked B0 logits")
        zero = torch.zeros_like(initial)
        bpp = physical[0] if self.mode != "E0_no_thermo" else zero
        stack = physical[1] if self.mode == "E2_bpp_stack" else (marginal_product(bpp) if self.mode == PRODUCT_MODE else zero)
        channels = torch.stack((initial, bpp, stack, distance, valid_mask.float()))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=embedding.is_cuda):
            residual = self.correction(channels).float()
        corrected = initial + residual
        corrected = (corrected + corrected.transpose(0, 1)) / 2
        corrected = corrected.masked_fill(~valid_mask, -100.0)
        return (corrected, initial, residual) if return_initial else corrected


def build_records(root, identifiers, manifest, features):
    records = {}
    for identifier in identifiers:
        row = manifest[identifier]
        sequence = row["sequence"]
        length = len(sequence)
        reference = pair_set_0b(pairs_from_str(row["eligible_pairs"]))
        original = pair_set_0b(pairs_from_str(row["original_pairs"]))
        cand_i, cand_j, labels = build_candidates(sequence, reference)
        valid = torch.zeros((length, length), dtype=torch.bool)
        valid[cand_i, cand_j] = True
        valid[cand_j, cand_i] = True
        positions = torch.arange(length, dtype=torch.float32)
        distance = (positions[:, None] - positions[None, :]).abs() / length
        cache_path = root / features[identifier]["cache_path"]
        if hashlib.sha256(cache_path.read_bytes()).hexdigest() != features[identifier]["cache_sha256"]:
            raise RuntimeError("Feature checksum mismatch: " + identifier)
        with np.load(cache_path, allow_pickle=False) as cached:
            upper_bpp = np.triu(cached["P"].astype(np.float32), 1)
            upper_stack = np.triu(cached["Q"].astype(np.float32), 1)
        if upper_bpp.shape != (length, length) or upper_stack.shape != (length, length):
            raise ValueError("Feature shape mismatch: " + identifier)
        physical = torch.from_numpy(np.stack((upper_bpp + upper_bpp.T, upper_stack + upper_stack.T)))
        physical *= valid.unsqueeze(0)
        records[identifier] = dict(sequence=sequence, reference=original, eligible=reference, original=original,
                                   family=row.get("family_id") or "unknown", cand_i=cand_i,
                                   cand_j=cand_j, labels=labels, valid=valid,
                                   distance=distance, physical=physical)
    return records


def model_inputs(record, embedding, device):
    return (embedding.to(device=device, dtype=torch.float32),
            record["physical"].to(device), record["distance"].to(device), record["valid"].to(device))


@torch.no_grad()
def evaluate(model, identifiers, embeddings, records, thresholds, device, output_path=None):
    model.eval()
    scores = {threshold: {"prf": [], "family": [], "original_prf": [], "original_family": []}
              for threshold in thresholds}
    output = output_path.open("w", encoding="utf8") if output_path else None
    started = time.time()
    try:
        for index, identifier in enumerate(identifiers):
            record = records[identifier]
            logits = model(*model_inputs(record, embeddings[identifier], device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation: " + identifier)
            predictions = decode_multi(logits.cpu().numpy(), record["sequence"], thresholds)
            for threshold, pairs in predictions.items():
                metrics = pair_prf(record["reference"], pairs)
                original_metrics = pair_prf(record["original"], pairs)
                scores[threshold]["prf"].append(metrics)
                scores[threshold]["family"].append((record["family"], metrics[2]))
                scores[threshold]["original_prf"].append(original_metrics)
                scores[threshold]["original_family"].append((record["family"], original_metrics[2]))
                if output:
                    output.write(json.dumps(dict(rna_id=identifier, sequence=record["sequence"],
                                                 family_id=record["family"], threshold=threshold,
                                                 reference_pairs=sorted(record["reference"]),
                                                 eligible_reference_pairs=sorted(record["eligible"]),
                                                 original_pairs=sorted(record["original"]),
                                                 full_reference_pairs=sorted(record["original"]),
                                                 predicted_pairs=pairs, exact_f1=metrics[2],
                                                 original_exact_f1=original_metrics[2])) + "\n")
            if (index + 1) % 300 == 0:
                print("EVAL", index + 1, "/", len(identifiers), "seconds", round(time.time() - started, 1), flush=True)
    finally:
        if output:
            output.close()
    return {threshold: dict(family_macro_exact_f1=family_macro(values["family"]),
                            rna_macro_exact_f1=rna_macro(values["prf"])[2],
                            original_family_macro_exact_f1=family_macro(values["original_family"]),
                            original_rna_macro_exact_f1=rna_macro(values["original_prf"])[2])
            for threshold, values in scores.items()}


def train_locked(args):
    root = Path(args.root).resolve()
    if root != ROOT.resolve():
        raise RuntimeError("--root must match this source checkout")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text())
    seed = args.seed if args.seed is not None else config["seed"]
    configuration_hash = digest(config)
    output_root = root / "outputs/thermo_fast"
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
            expected_predictions = {"family-dev_predictions.jsonl": summary["dev_count"], "VL0_predictions.jsonl": summary["VL0_count"]}
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
    train_ids = [identifier for identifier in split["fit_train"] if features.get(identifier, {}).get("status") == "success"]
    dev_ids = [identifier for identifier in split["family_dev"] if features.get(identifier, {}).get("status") == "success"]
    vl_ids = [identifier for identifier in split["VL0"] if features.get(identifier, {}).get("status") == "success"]
    if args.smoke:
        train_ids = train_ids[:32]
        dev_ids = dev_ids[:16]
        vl_ids = []
    if not train_ids or not dev_ids or (not args.smoke and not vl_ids):
        raise RuntimeError("Empty required shared successful cohort")
    identifiers = train_ids + dev_ids + vl_ids
    records = build_records(root, identifiers, manifest, features)
    embeddings = {}
    for name, requested in (("fit_train", train_ids), ("family_dev", dev_ids), ("VL0", vl_ids)):
        if requested:
            cached = load_embeddings(root, name)
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
                  VL0_count=len(vl_ids), train_ids_hash=digest(train_ids), dev_ids_hash=digest(dev_ids),
                  encoder_frozen=True, loaded_supervised_checkpoint=False)
    save_json(run_path / "status.json", status)
    save_json(run_path / "config.json", config)
    save_json(run_path / "resolved_config.json", dict(config, seed=seed, mode=args.mode, smoke=args.smoke))
    save_json(run_path / "cohort.json", dict(fit_train=train_ids, family_dev=dev_ids, VL0=vl_ids))
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
                  source_sha256=hashlib.sha256(source_bytes).hexdigest(), audit_path="outputs/thermo_fast/split_audit.json",
                  cache_manifest_path="outputs/thermo_fast/thermo_cache_manifest.json"))
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
        if checkpoint.get("dev_ids_hash", digest(dev_ids)) != digest(dev_ids) or checkpoint.get("VL0_ids_hash", digest(vl_ids)) != digest(vl_ids):
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
            threshold = max(thresholds, key=lambda value: evaluation[value]["family_macro_exact_f1"])
            score = evaluation[threshold]["family_macro_exact_f1"]
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
                              VL0_ids_hash=digest(vl_ids), input_provenance_hash=input_provenance_hash)
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
                          best_family_macro_exact_f1=best_score, elapsed_training_seconds=training_seconds,
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
    dev_metrics = evaluate(model, dev_ids, embeddings, records, [best_threshold], device, run_path / "family-dev_predictions.jsonl")[best_threshold]
    vl_metrics = evaluate(model, vl_ids, embeddings, records, [best_threshold], device, run_path / "VL0_predictions.jsonl")[best_threshold] if vl_ids else None
    evaluation_seconds += time.time() - started
    summary = dict(status="completed", mode=args.mode, seed=seed, smoke=args.smoke,
                   best_epoch=best_epoch, threshold=best_threshold, epochs_completed=completed_epoch,
                   **dev_metrics, VL0_exact_f1=vl_metrics["rna_macro_exact_f1"] if vl_metrics else None,
                   VL0_original_exact_f1=vl_metrics["original_rna_macro_exact_f1"] if vl_metrics else None,
                   parameter_count=parameter_count, initialization_hash=initialization_hash,
                   config_hash=configuration_hash, training_seconds=training_seconds,
                   evaluation_seconds=evaluation_seconds, gradients_verified=gradients_verified,
                   train_count=len(train_ids), dev_count=len(dev_ids), VL0_count=len(vl_ids),
                   train_ids_hash=digest(train_ids), dev_ids_hash=digest(dev_ids),
                   VL0_ids_hash=digest(vl_ids),
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
    seed = args.seed if args.seed is not None else config["seed"]
    run_path = root / "outputs/thermo_fast" / ("smoke" if args.smoke else "") / args.mode / f"seed{seed}"
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


def pipeline(args):
    root = Path(args.root).resolve()
    if root != ROOT.resolve():
        raise RuntimeError("--root must match this source checkout")
    output = root / "outputs/thermo_fast"
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "pipeline_logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(root / ".tmp"),
               TORCH_HOME=str(root / ".cache/torch"), CUDA_CACHE_PATH=str(root / ".cache/cuda"),
               XDG_CACHE_HOME=str(root / ".cache"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for name in (".tmp", ".cache/torch", ".cache/cuda"):
        (root / name).mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline_status.json"
    started = time.time()

    def stage(name, script, *arguments):
        command = [sys.executable, str(root / script), "--root", str(root), *map(str, arguments)]
        stage_started = time.time()
        state = dict(status="running", stage=name, pid=os.getpid(), command=command,
                     stage_start_unix=stage_started, pipeline_start_unix=started)
        save_json(status_path, state)
        print("STAGE_START", name, flush=True)
        with (logs / (name + ".log")).open("a") as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            save_json(status_path, state)
            while child.poll() is None:
                time.sleep(15)
                state["elapsed_seconds"] = time.time() - stage_started
                save_json(status_path, state)
        state.update(status="completed" if child.returncode == 0 else "failed",
                     returncode=child.returncode, elapsed_seconds=time.time() - stage_started)
        save_json(status_path, state)
        with (output / "pipeline_stages.jsonl").open("a") as handle:
            handle.write(json.dumps(state) + "\n")
        print("STAGE_END", name, child.returncode, round(state["elapsed_seconds"], 1), flush=True)
        with (root / "PROGRESS.md").open("a") as progress:
            progress.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC: {name}: returncode={child.returncode}, seconds={state['elapsed_seconds']:.1f}\n")
        if child.returncode:
            raise RuntimeError("Stage failed: " + name + "; inspect " + str(logs / (name + ".log")))

    try:
        provenance = output / "provenance"
        provenance.mkdir(exist_ok=True)
        code_files = sorted((root / "src").glob("*.py")) + sorted((root / "scripts").glob("*.py")) + [root / args.config]
        source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files}
        save_json(provenance / "implementation_manifest.json", dict(code_sha256=source_hashes,
                   adaptation="Original B0 PairHead, pair_bce_loss, candidates, greedy decoder and metrics are copied unchanged. Only the three ThermoFusion scripts and config are added.",
                   loss="Original per-RNA unweighted candidate BCE; lr=1e-4, weight_decay=1e-4 and accumulation=4 reused from original current_protocol.json.",
                   selection="family-dev only, equal-family macro exact F1, lowest threshold on ties, earliest trained epoch on ties; maximum 10 epochs and identical patience 3 for every mode.",
                   checkpoint_provenance="No old supervised checkpoint copied or loaded; identical random initialization by seed; encoder embeddings are generic pretrained RiNALMo-Giga.",
                   test_policy="TS0 and bpRNA-new not inferred or scored in FAST-v1."))
        save_json(output / "environment_audit.json", dict(python=sys.version, executable=sys.executable,
                   torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                   cuda_available=torch.cuda.is_available(), bf16_supported=torch.cuda.is_bf16_supported(),
                   gpu=torch.cuda.get_device_name(0), project_realpath=str(root),
                   expected_vienna="2.6.4", daemon="nohup_isolated_process", source_project="/root/autodl-tmp/rna_stem_supervision", vienna_actual=__import__("RNA").__version__))
        stage("preflight", "scripts/preflight.py")
        stage("contracts", "tests/test_contract.py")
        stage("cache_thermo", "scripts/cache_thermo.py", "--split", "fit-train,family-dev,VL0", "--features", "bpp,stack", "--workers", 8)
        for mode in MODES:
            stage("smoke_" + mode, "scripts/run_thermo_fast.py", "--config", args.config, "--mode", mode, "--seed", 17, "--smoke")
        smoke_summaries = [json.loads((output / "smoke" / mode / "seed17/summary.json").read_text()) for mode in MODES]
        if len({value["initialization_hash"] for value in smoke_summaries}) != 1 or len({value["parameter_count"] for value in smoke_summaries}) != 1:
            raise AssertionError("Smoke initialization/parameter equality failed")
        if not all(value["gradients_verified"] for value in smoke_summaries):
            raise AssertionError("Smoke gradients failed")
        save_json(output / "smoke_summary.json", dict(passed=True, runs=smoke_summaries,
                   interpretation="Program validation only; small-sample F1 is not scientific evidence."))
        for mode in MODES[:2]:
            stage(mode + "_seed17", "scripts/run_thermo_fast.py", "--config", args.config, "--mode", mode, "--seed", 17)
        stage("evaluate_seed17", "scripts/eval_thermo_fast.py", "--compare", ",".join(MODES), "--seeds", "17")
        decision = json.loads((output / "decision.json").read_text())
        run_stack = decision.get("run_E2", False)
        seeds = [17]
        if run_stack:
            stage("E2_bpp_stack_seed17", "scripts/run_thermo_fast.py", "--config", args.config, "--mode", MODES[2], "--seed", 17)
            confirmation_seeds = [23, 41]
        else:
            confirmation_seeds = decision.get("seeds_to_run", [23])
        for seed in confirmation_seeds:
            if seed == 17:
                continue
            for mode in MODES if run_stack else MODES[:2]:
                stage(mode + f"_seed{seed}", "scripts/run_thermo_fast.py", "--config", args.config, "--mode", mode, "--seed", seed)
            seeds.append(seed)
            stage("evaluate_seed" + str(seed), "scripts/eval_thermo_fast.py", "--compare", ",".join(MODES), "--seeds", ",".join(map(str, seeds)))
        stage("final_analysis", "scripts/eval_thermo_fast.py", "--compare", ",".join(MODES), "--seeds", ",".join(map(str, seeds)))
        decision = json.loads((output / "decision.json").read_text())
        if decision.get("Q_retention") == "retain_candidate":
            for seed in (17, 23, 41):
                stage(PRODUCT_MODE + f"_seed{seed}", "scripts/run_thermo_fast.py", "--config", args.config, "--mode", PRODUCT_MODE, "--seed", seed)
            stage("product_analysis", "scripts/eval_thermo_fast.py", "--compare", ",".join((*MODES, PRODUCT_MODE)), "--seeds", "17,23,41")
        stage("validate_and_report", "scripts/finalize_results.py")
        save_json(status_path, dict(status="completed", seeds=seeds, E2_executed=run_stack,
                   seconds=time.time() - started, test_data_evaluated=False))
        print("PIPELINE_COMPLETED", seeds, "E2", run_stack, flush=True)
    except Exception:
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        state.update(status="failed", error=traceback.format_exc())
        save_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default="configs/thermo_fast.yaml")
    parser.add_argument("--mode", choices=(*MODES, PRODUCT_MODE))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.pipeline:
        (ROOT / "outputs/thermo_fast").mkdir(parents=True, exist_ok=True)
        with (ROOT / "outputs/thermo_fast/.pipeline.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline(args)
        return
    if args.mode is None:
        parser.error("--mode is required unless --pipeline is selected")
    train(args)


if __name__ == "__main__":
    main()
