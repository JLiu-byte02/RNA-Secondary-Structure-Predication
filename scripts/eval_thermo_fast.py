import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXCLUDED_FAMILIES = (None, "unknown", "NA", "")
POSITIVE_GAIN = 0.015
CONCENTRATION_LIMIT = 0.5


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_pairs(values, sequence_length, context):
    pairs = set()
    for pair in values:
        if len(pair) != 2:
            raise ValueError(f"Invalid pair in {context}: {pair}")
        first, second = pair
        if not isinstance(first, int) or not isinstance(second, int):
            raise ValueError(f"Noninteger pair in {context}: {pair}")
        if not 0 <= first < second < sequence_length:
            raise ValueError(f"Invalid 0-based coordinates in {context}: {pair}")
        pairs.add((first, second))
    return pairs


def load_predictions(path):
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            rna_id = str(record["rna_id"])
            if rna_id in records:
                raise ValueError(f"Duplicate rna_id {rna_id} in {path}")
            sequence = record["sequence"]
            context = f"{path}:{line_number} ({rna_id})"
            record["reference"] = normalize_pairs(record["reference_pairs"], len(sequence), context)
            record["predicted"] = normalize_pairs(record["predicted_pairs"], len(sequence), context)
            full_values = record.get("full_reference_pairs", record.get("original_pairs"))
            if full_values is None:
                raise ValueError(f"Missing full original reference labels in {context}")
            record["full_reference"] = normalize_pairs(full_values, len(sequence), context)
            record["eligible_reference"] = normalize_pairs(record["eligible_reference_pairs"], len(sequence), context)
            if record["reference"] != record["full_reference"]:
                raise ValueError("Primary truth must be the complete original truth")
            if "original_pairs" in record and "full_reference_pairs" in record:
                original = normalize_pairs(record["original_pairs"], len(sequence), context)
                if original != record["full_reference"]:
                    raise ValueError(f"Full-truth aliases disagree in {context}")
            if not record["eligible_reference"].issubset(record["full_reference"]):
                raise ValueError(f"Eligible labels are not a subset of original truth in {context}")
            records[rna_id] = record
    if not records:
        raise ValueError(f"Empty prediction file: {path}")
    return records


def metric_bundle(records, reference_key, pair_prf, family_macro, rna_macro):
    triples = []
    grouped = defaultdict(list)
    family_values = []
    for record in records.values():
        precision, recall, f1 = pair_prf(record[reference_key], record["predicted"])
        triples.append((precision, recall, f1))
        family = record.get("family_id")
        family_values.append((family, f1))
        if family not in EXCLUDED_FAMILIES:
            grouped[str(family)].append(f1)
    families = {family: {"count": len(values), "f1": sum(values) / len(values)} for family, values in grouped.items()}
    family_score = family_macro(family_values) if families else None
    rna_score = rna_macro(triples)[2]
    if family_score is not None and not math.isfinite(family_score):
        raise ValueError("Nonfinite family macro F1")
    if not math.isfinite(rna_score):
        raise ValueError("Nonfinite RNA macro F1")
    return {"family_macro": family_score, "rna_macro": rna_score, "families": families}


def require_matched_cohort(records, baseline, mode, seed):
    if set(records) != set(baseline):
        missing = sorted(set(baseline) - set(records))[:5]
        extra = sorted(set(records) - set(baseline))[:5]
        raise ValueError(f"Cohort mismatch {mode}/seed{seed}: missing={missing}, extra={extra}")
    for rna_id, record in records.items():
        other = baseline[rna_id]
        for field in ("sequence", "reference", "full_reference", "eligible_reference"):
            if record[field] != other[field]:
                raise ValueError(f"{field} mismatch for {rna_id}: {mode}/seed{seed}")
        if record.get("family_id") != other.get("family_id"):
            raise ValueError(f"family_id mismatch for {rna_id}: {mode}/seed{seed}")


def stem_statistics(record, predicted, maximal_positive_stems):
    stems = maximal_positive_stems(record["sequence"], record["reference"], min_stem=3)
    complete = 0
    recovered_pairs = 0
    stem_pairs = 0
    for first, second, length in stems:
        reference = {(first + offset, second - offset) for offset in range(length)}
        recovered = len(reference & predicted)
        complete += int(reference.issubset(predicted))
        recovered_pairs += recovered
        stem_pairs += len(reference)
    return len(stems), complete, recovered_pairs, stem_pairs


def scan_statuses(output_root):
    statuses = []
    for run_directory in sorted(output_root.glob("*/seed*")):
        if not run_directory.is_dir():
            continue
        entry = {"mode": run_directory.parent.name, "seed_directory": run_directory.name}
        for filename in ("status.json", "summary.json"):
            path = run_directory / filename
            if path.exists():
                try:
                    entry[filename[:-5]] = read_json(path)
                except (OSError, ValueError) as error:
                    entry[filename[:-5]] = {"read_error": str(error)}
        statuses.append(entry)
    return statuses


def make_decision(runs):
    evidence = {}
    for seed in sorted({seed for _, seed in runs}):
        baseline = runs.get(("E0_no_thermo", seed))
        bpp = runs.get(("E1_bpp", seed))
        if baseline is None or bpp is None:
            continue
        baseline_metric = baseline["metrics"]
        bpp_metric = bpp["metrics"]
        if baseline_metric["family_macro"] is None or bpp_metric["family_macro"] is None:
            continue
        positive_deltas = sorted((bpp_metric["families"][family]["f1"] - values["f1"] for family, values in baseline_metric["families"].items()), reverse=True)
        positive_deltas = [delta for delta in positive_deltas if delta > 0]
        positive_total = sum(positive_deltas)
        concentration = sum(positive_deltas[:5]) / positive_total if positive_total else 1.0
        evidence[str(seed)] = {
            "family_gain": bpp_metric["family_macro"] - baseline_metric["family_macro"],
            "rna_gain": bpp_metric["rna_macro"] - baseline_metric["rna_macro"],
            "known_families": len(baseline_metric["families"]),
            "families_improved": len(positive_deltas),
            "top5_positive_delta_fraction": concentration,
        }
    decision = {"decision": "weak", "run_E2": False, "seeds_to_run": [], "stage": "incomplete", "supporting_metrics": evidence, "gate_rule": {"family_gain_at_least": POSITIVE_GAIN, "rna_gain_strictly_positive": True, "top5_positive_delta_fraction_at_most": CONCENTRATION_LIMIT, "excluded_family_ids": list(EXCLUDED_FAMILIES)}, "scientific_significance_claim": False}
    initial = evidence.get("17")
    if initial is None:
        decision["reason"] = "Completed matched seed17 E0/E1 predictions are required."
        return decision
    positive = initial["family_gain"] >= POSITIVE_GAIN - 1e-12 and initial["rna_gain"] > 0 and initial["top5_positive_delta_fraction"] <= CONCENTRATION_LIMIT + 1e-12
    if positive:
        decision.update(decision="positive", run_E2=True, seeds_to_run=[seed for seed in (23, 41) if str(seed) not in evidence], stage="confirmation", reason="Seed17 passes the predeclared development gate; confirm with matched seeds23/41 and conditional E2.")
        if "23" in evidence and "41" in evidence:
            stable = all(values["family_gain"] > 0 and values["rna_gain"] > 0 for seed, values in evidence.items() if seed in ("17", "23", "41"))
            decision["E1_cross_seed_consistent"] = stable
            decision["stage"] = "confirmation_complete"
            if not stable:
                decision.update(decision="weak", run_E2=False, reason="The seed17 signal is not consistently positive in matched confirmation seeds; stop expanding the method.")
    elif initial["family_gain"] < 0:
        confirmation = evidence.get("23")
        if confirmation is None:
            decision.update(decision="negative", seeds_to_run=[23], stage="needs_repeat", reason="Seed17 declines; check alignment/settings and repeat E0/E1 with seed23 before stopping.")
        elif confirmation["family_gain"] < 0:
            decision.update(decision="negative", stage="stop", reason="E1 declines in both seeds17/23; stop additional modules and investigate the recorded checks.")
        else:
            decision.update(decision="weak", stage="stop", reason="The negative seed17 result does not repeat; evidence is mixed and does not justify E2.")
    else:
        decision.update(seeds_to_run=[23] if "23" not in evidence else [], stage="needs_repeat" if "23" not in evidence else "stop", reason="Seed17 does not meet the positive gate; inspect family/pair changes and use at most seed23 for confirmation.")
    stack_evidence = {}
    for seed in (17, 23, 41):
        bpp = runs.get(("E1_bpp", seed))
        stack = runs.get(("E2_bpp_stack", seed))
        if bpp and stack:
            stack_evidence[str(seed)] = {"family_gain_over_E1": stack["metrics"]["family_macro"] - bpp["metrics"]["family_macro"], "rna_gain_over_E1": stack["metrics"]["rna_macro"] - bpp["metrics"]["rna_macro"]}
    decision["Q_supporting_metrics"] = stack_evidence
    decision["Q_retention"] = "pending" if len(stack_evidence) < 3 else ("retain_candidate" if all(values["family_gain_over_E1"] > 0 and values["rna_gain_over_E1"] > 0 for values in stack_evidence.values()) else "prefer_E1")
    decision["missing_E2_seeds"] = [seed for seed in (17, 23, 41) if ("E2_bpp_stack", seed) not in runs] if decision["run_E2"] else []
    return decision


def render_report(output_root, split, rows, decision, statuses, issues):
    audit_path = output_root / "split_audit.json"
    cache_path = output_root / "thermo_cache_manifest.json"
    audit = read_json(audit_path) if audit_path.exists() else {"status": "missing"}
    cache = read_json(cache_path) if cache_path.exists() else {"status": "missing"}
    audit = {key: audit.get(key) for key in ("passed", "counts", "known_family_counts", "fit_family_overlap", "unique_sequences", "embedding_dimension", "embedding_special_tokens", "embedding_sha256", "manifest_sha256", "split_ids_sha256", "label_totals", "decoder_unreachable_original_truth_fraction", "supervised_initialization", "label_policy", "errors")}
    cache = {**{key: cache.get(key) for key in ("viennarna_version", "params", "features", "stack_cutoff", "matrix_layout", "stack_api", "stack_api_source", "bpp_api_source", "coverage", "duration_seconds", "cohort_policy")}, "unique_sequence_failures": len(cache.get("failures", [])), "failure_examples": cache.get("failures", [])[:10]}
    lines = ["# ThermoFusion fast-v1 development results", "", f"Evaluated split: `{split}`. This report records completed and pending development experiments; it makes no statistical significance or final-test claim.", "", "Primary exact F1 uses `reference_pairs`: the full unfiltered original truth. Full-original-truth F1 uses `full_reference_pairs` / `original_pairs`; no unsupported truth is silently relabeled. Family macro gives equal weight to known families and excludes null, unknown, NA and empty family IDs. RNA macro includes every RNA in the common prediction cohort.", "", "| Mode | Seed | RNAs | Family macro | RNA macro | Full-truth family macro | Full-truth RNA macro | Eligible labels removed / full labels |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["mode"], str(row["seed"]), str(row["rna_count"])]
        values.extend("unavailable" if row[key] is None else f"{row[key]:.6f}" for key in ("family_macro_exact_f1", "rna_macro_exact_f1", "full_original_family_macro_exact_f1", "full_original_rna_macro_exact_f1", "excluded_truth_fraction"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Across completed seeds", "", "| Mode | Seeds | Family exact F1 mean ± sample SD | RNA exact F1 mean ± sample SD | Mean family gain over matched E0 | VL0 auxiliary mean |", "|---|---|---:|---:|---:|---:|"])
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        family_values = [row["family_macro_exact_f1"] for row in group]
        rna_values = [row["rna_macro_exact_f1"] for row in group]
        gains = [row["family_gain_vs_E0"] for row in group if row.get("family_gain_vs_E0") is not None]
        vl_values = [row["VL0_exact_f1"] for row in group if row.get("VL0_exact_f1") is not None]
        family_sd = f"{statistics.stdev(family_values):.6f}" if len(group) > 1 else "not estimated"
        rna_sd = f"{statistics.stdev(rna_values):.6f}" if len(group) > 1 else "not estimated"
        lines.append(f"| {mode} | {','.join(str(row['seed']) for row in group)} | {statistics.fmean(family_values):.6f} ± {family_sd} | {statistics.fmean(rna_values):.6f} ± {rna_sd} | {statistics.fmean(gains):.6f} | {statistics.fmean(vl_values):.6f} |" if gains and vl_values else f"| {mode} | {','.join(str(row['seed']) for row in group)} | {statistics.fmean(family_values):.6f} ± {family_sd} | {statistics.fmean(rna_values):.6f} ± {rna_sd} | unavailable | unavailable |")
    correction_path = output_root / "pair_stem_corrections.csv"
    if correction_path.exists():
        with correction_path.open(newline="", encoding="utf8") as handle:
            corrections = list(csv.DictReader(handle))
        lines.extend(["", "## Paired structure repair totals", "", "Counts pool the same-seed RNA comparisons across completed seeds. A true added pair is absent from E0 and present in the reference; a false deleted pair is an E0 error removed by the compared mode.", "", "| Mode | True pairs added | False pairs deleted | False pairs added | True pairs deleted | Complete stems recovered | E0 complete stems recovered | Reference stems |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for mode in sorted({record["mode"] for record in corrections}):
            group = [record for record in corrections if record["mode"] == mode]
            fields = ("eligible_true_added", "eligible_false_deleted", "eligible_false_added", "eligible_true_deleted", "complete_stems_recovered", "E0_complete_stems_recovered", "reference_stems")
            totals = [sum(int(record[field]) for record in group) for field in fields]
            lines.append("| " + mode + " | " + " | ".join(map(str, totals)) + " |")
    lines.extend(["", f"Development decision: **{decision['decision']}** ({decision['stage']}). {decision['reason']}", f"Run E2: {decision['run_E2']}; missing E0/E1 confirmation seeds: {decision['seeds_to_run']}; missing conditional E2 seeds: {decision.get('missing_E2_seeds', [])}.", "", "The predeclared positive gate is family-macro gain ≥0.015, positive RNA-macro gain, and at most 50% of summed positive per-family gains concentrated in the five largest gains. This heuristic allocates compute; it is not a significance test. VL0 never determines the gate. Decisions use full original truth, while full-truth metrics are reported alongside.", "", f"Q retention: `{decision.get('Q_retention', 'pending')}`. Any eventual joint-statistic claim also requires the real-Q versus P_outer×P_inner control with identical architecture and budget.", "", "Pair corrections in pair_stem_corrections.csv are relative to the same-seed E0 predictions and include both eligible and full-original truth counts. Stem statistics reuse maximal_positive_stems(min_stem=3) with full original reference labels; partial recovery is the recovered reference-stem pair fraction. CSV files identify that label choice explicitly.", "", "No TS0 or bpRNA-new prediction files are read by this evaluator. Earlier project test-set exposure prevents a claim that these datasets never influenced method design. Strict unseen-family claims also require verified checkpoint and split provenance.", "", "## Runtime and current run statuses", "", "```json", json.dumps(statuses, indent=2, ensure_ascii=False), "```", "", "## Split and checkpoint provenance audit", "", "```json", json.dumps(audit, indent=2, ensure_ascii=False), "```", "", "## Thermodynamic cache and coverage", "", "```json", json.dumps(cache, indent=2, ensure_ascii=False), "```", "", "## Issues and pending evidence", ""])
    lines.extend([f"- {issue}" for issue in issues] or ["- No prediction-contract or common-cohort issues detected."])
    lines.extend(["", "## Implementation and reproducibility", "", "The designated directory was initially empty. Necessary original data manifests, generic RiNALMo embeddings and unchanged B0 source modules were copied from /root/autodl-tmp/rna_stem_supervision with authorization; that source project was not modified. No full-TR0 or other supervised B0 checkpoint was loaded. The head and zero-initialized residual CNN train jointly from identical random initialization per seed.", "", "ViennaRNA 2.6.4 was installed in the project-local .venv. Its Python wrapper lacks stack_prob (the wrapper arrived in 2.7.0); cache_thermo.py calls the exported 2.6.4 vrna_stack_prob C function, stops at its i=0 sentinel, converts coordinates once and releases the buffer with the same Linux allocator. One reference partition function produces both P and genuine Q, with Q checked against both pair marginals. This is not a product approximation.", "", "The original B0 unweighted candidate BCE and mature lr=1e-4, weight decay=1e-4, accumulation=4 were reused uniformly. All modes have 186306 parameters and maximum 10 epochs with patience 3; actual stopping epochs can differ under this same rule. Thresholds 0.1–0.9 are selected only on family-dev family-macro F1, with the lowest threshold and earliest trained epoch breaking ties. VL0 is scored only at that selected threshold.", "", "Reproduce from this project directory: .venv/bin/python -u scripts/run_thermo_fast.py --pipeline. The pipeline is idempotent for completed matching configurations and resumes last.pt after interruptions. Long jobs run as isolated nohup processes (all paths under Astra). Check pipeline_status.json, pipeline.log and pipeline_logs/*.log. Checkpoints, per-epoch logs and both family-dev/VL0 predictions are saved under each mode/seed directory.", "", "## Next-stage limits", "", "A stable positive BPP result supports further evaluation under a frozen method. Additional Q gains must be consistent across all three seeds before Q is retained; a claim of independent joint-statistic value still requires the real-Q versus P_outer×P_inner control. Full-TR0 retraining and once-only TS0/bpRNA-new testing are the subsequent phase described in README section 7, not completed by this fast-v1 report. Negative or mixed E1 evidence does not justify expanding to nine perturbed thermodynamic maps.", "", "Detailed gate and per-seed evidence: decision.json. Complete per-RNA audit records and cache manifests remain in split_audit.json and thermo_cache_manifest.json; this report prints bounded summaries. Family differences and pair/stem repairs are in the three CSV files. Historical B0 results are not a fair same-protocol development baseline because the old checkpoint/threshold selection differed; E0 is the controlled baseline here.", ""])
    (output_root / "README_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("family-dev", "VL0"), default="family-dev")
    parser.add_argument("--compare", default="E0_no_thermo,E1_bpp,E2_bpp_stack")
    parser.add_argument("--seeds", default="17")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from src.metrics import family_macro, pair_prf, rna_macro
    from src.stem_candidates import maximal_positive_stems

    modes = list(dict.fromkeys(mode.strip() for mode in args.compare.split(",") if mode.strip()))
    allowed = {"E0_no_thermo", "E1_bpp", "E2_bpp_stack", "E3_bpp_product"}
    if not modes or set(modes) - allowed:
        parser.error("--compare must name E0_no_thermo, E1_bpp and/or E2_bpp_stack")
    seeds = sorted(set(int(value.strip()) for value in args.seeds.split(",") if value.strip()))
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    output_root = root / "outputs" / "thermo_fast"
    output_root.mkdir(parents=True, exist_ok=True)
    runs = {}
    issues = []
    summary_rows = []
    for mode in modes:
        for seed in seeds:
            run_directory = output_root / mode / f"seed{seed}"
            summary_path = run_directory / "summary.json"
            if not summary_path.exists():
                issues.append(f"{mode}/seed{seed}: no completed summary; omitted from metrics.")
                continue
            summary = read_json(summary_path)
            if summary.get("status") != "completed":
                issues.append(f"{mode}/seed{seed}: status {summary.get('status', 'unspecified')}; omitted from metrics.")
                continue
            if summary.get("mode", mode) != mode or int(summary.get("seed", seed)) != seed:
                raise ValueError(f"Run identity mismatch in {summary_path}")
            for field in ("initialization_hash", "config_hash", "parameter_count"):
                if summary.get(field) is None:
                    issues.append(f"{mode}/seed{seed}: missing {field}; the corresponding fair-comparison provenance remains unverified.")
            prediction_path = run_directory / f"{args.split}_predictions.jsonl"
            if not prediction_path.exists():
                issues.append(f"{mode}/seed{seed}: completed summary but missing {prediction_path.name}; omitted.")
                continue
            records = load_predictions(prediction_path)
            metrics = metric_bundle(records, "reference", pair_prf, family_macro, rna_macro)
            full_metrics = metric_bundle(records, "full_reference", pair_prf, family_macro, rna_macro)
            runs[(mode, seed)] = {"records": records, "metrics": metrics, "full_metrics": full_metrics, "summary": summary}
            full_count = sum(len(record["full_reference"]) for record in records.values())
            excluded_count = sum(len(record["full_reference"] - record["eligible_reference"]) for record in records.values())
            row = {"mode": mode, "seed": seed, "split": args.split, "rna_count": len(records), "known_family_count": len(metrics["families"]), "unknown_family_rna_count": sum(record.get("family_id") in EXCLUDED_FAMILIES for record in records.values()), "family_macro_exact_f1": metrics["family_macro"], "rna_macro_exact_f1": metrics["rna_macro"], "full_original_family_macro_exact_f1": full_metrics["family_macro"], "full_original_rna_macro_exact_f1": full_metrics["rna_macro"], "full_truth_pair_count": full_count, "excluded_truth_pair_count": excluded_count, "excluded_truth_fraction": excluded_count / full_count if full_count else 0.0, "VL0_exact_f1": summary.get("VL0_exact_f1"), "primary_label_policy": "full_original_reference_pairs", "full_truth_label_policy": "full_reference_pairs_or_original_pairs"}
            for field in ("best_epoch", "threshold", "epochs_completed", "parameter_count", "initialization_hash", "config_hash", "training_seconds", "evaluation_seconds"):
                row[field] = summary.get(field)
            runs[(mode, seed)]["summary_row"] = row
            summary_rows.append(row)
    family_rows = []
    correction_rows = []
    for (mode, seed), run in runs.items():
        baseline = runs.get(("E0_no_thermo", seed))
        if baseline is None:
            issues.append(f"{mode}/seed{seed}: same-seed E0 unavailable; paired comparisons omitted.")
            continue
        require_matched_cohort(run["records"], baseline["records"], mode, seed)
        for field in ("config_hash", "input_provenance_hash", "train_ids_hash", "dev_ids_hash"):
            if run["summary"].get(field) != baseline["summary"].get(field):
                raise ValueError(f"Comparison provenance mismatch: {field}")
        if run["summary"].get("initialization_hash") != baseline["summary"].get("initialization_hash"):
            raise ValueError(f"Initialization mismatch for {mode}/seed{seed}")
        if run["summary"].get("parameter_count") != baseline["summary"].get("parameter_count"):
            raise ValueError(f"Parameter-count mismatch for {mode}/seed{seed}")
        for field, metric_key in (("family_gain_vs_E0", "family_macro"), ("rna_gain_vs_E0", "rna_macro")):
            current = run["metrics"][metric_key]
            previous = baseline["metrics"][metric_key]
            run["summary_row"][field] = current - previous if current is not None and previous is not None else None
        for field, metric_key in (("full_original_family_gain_vs_E0", "family_macro"), ("full_original_rna_gain_vs_E0", "rna_macro")):
            current = run["full_metrics"][metric_key]
            previous = baseline["full_metrics"][metric_key]
            run["summary_row"][field] = current - previous if current is not None and previous is not None else None
        for family, values in run["metrics"]["families"].items():
            other = baseline["metrics"]["families"][family]
            family_rows.append({"mode": mode, "seed": seed, "split": args.split, "family_id": family, "rna_count": values["count"], "exact_f1": values["f1"], "E0_exact_f1": other["f1"], "delta_exact_f1": values["f1"] - other["f1"], "full_original_exact_f1": run["full_metrics"]["families"][family]["f1"], "E0_full_original_exact_f1": baseline["full_metrics"]["families"][family]["f1"], "full_original_delta_exact_f1": run["full_metrics"]["families"][family]["f1"] - baseline["full_metrics"]["families"][family]["f1"]})
        for rna_id, record in run["records"].items():
            other = baseline["records"][rna_id]
            added = record["predicted"] - other["predicted"]
            deleted = other["predicted"] - record["predicted"]
            row = {"mode": mode, "seed": seed, "split": args.split, "rna_id": rna_id, "family_id": record.get("family_id"), "stem_label_policy": "full_original_reference_pairs", "excluded_truth_pair_count": len(record["full_reference"] - record["eligible_reference"])}
            for prefix, reference_key in (("eligible", "eligible_reference"), ("full_original", "full_reference")):
                truth = record[reference_key]
                row.update({f"{prefix}_true_added": len(added & truth), f"{prefix}_false_deleted": len(deleted - truth), f"{prefix}_false_added": len(added - truth), f"{prefix}_true_deleted": len(deleted & truth)})
            total, recovered, partial_pairs, stem_pairs = stem_statistics(record, record["predicted"], maximal_positive_stems)
            _, baseline_recovered, baseline_partial, _ = stem_statistics(record, other["predicted"], maximal_positive_stems)
            row.update(reference_stems=total, complete_stems_recovered=recovered, E0_complete_stems_recovered=baseline_recovered, complete_stem_delta=recovered - baseline_recovered, reference_stem_pair_count=stem_pairs, recovered_stem_pair_count=partial_pairs, E0_recovered_stem_pair_count=baseline_partial, partial_stem_pair_fraction=partial_pairs / stem_pairs if stem_pairs else None, E0_partial_stem_pair_fraction=baseline_partial / stem_pairs if stem_pairs else None)
            correction_rows.append(row)
    summary_fields = ["mode", "seed", "split", "rna_count", "known_family_count", "unknown_family_rna_count", "family_macro_exact_f1", "rna_macro_exact_f1", "family_gain_vs_E0", "rna_gain_vs_E0", "full_original_family_macro_exact_f1", "full_original_rna_macro_exact_f1", "full_original_family_gain_vs_E0", "full_original_rna_gain_vs_E0", "VL0_exact_f1", "full_truth_pair_count", "excluded_truth_pair_count", "excluded_truth_fraction", "best_epoch", "threshold", "epochs_completed", "parameter_count", "initialization_hash", "config_hash", "training_seconds", "evaluation_seconds", "primary_label_policy", "full_truth_label_policy"]
    family_fields = ["mode", "seed", "split", "family_id", "rna_count", "exact_f1", "E0_exact_f1", "delta_exact_f1", "full_original_exact_f1", "E0_full_original_exact_f1", "full_original_delta_exact_f1"]
    correction_fields = ["mode", "seed", "split", "rna_id", "family_id", "eligible_true_added", "eligible_false_deleted", "eligible_false_added", "eligible_true_deleted", "full_original_true_added", "full_original_false_deleted", "full_original_false_added", "full_original_true_deleted", "excluded_truth_pair_count", "reference_stems", "complete_stems_recovered", "E0_complete_stems_recovered", "complete_stem_delta", "reference_stem_pair_count", "recovered_stem_pair_count", "E0_recovered_stem_pair_count", "partial_stem_pair_fraction", "E0_partial_stem_pair_fraction", "stem_label_policy"]
    if args.split != "family-dev":
        write_csv(output_root / f"{args.split}_summary.csv", summary_rows, summary_fields)
        write_csv(output_root / f"{args.split}_family_delta.csv", family_rows, family_fields)
        write_csv(output_root / f"{args.split}_pair_stem_corrections.csv", correction_rows, correction_fields)
        print(json.dumps({"split": args.split, "completed_runs": len(runs), "decision_unchanged": True, "issues": issues}, ensure_ascii=False))
        return
    write_csv(output_root / "development_summary.csv", summary_rows, summary_fields)
    write_csv(output_root / "family_delta.csv", family_rows, family_fields)
    write_csv(output_root / "pair_stem_corrections.csv", correction_rows, correction_fields)
    decision = make_decision(runs)
    write_json(output_root / "decision.json", decision)
    for filename in ("split_audit.json", "thermo_cache_manifest.json"):
        if not (output_root / filename).exists():
            issues.append(f"Missing {filename}; split/checkpoint provenance or thermodynamic coverage cannot be verified from this report.")
    render_report(output_root, args.split, summary_rows, decision, scan_statuses(output_root), issues)
    print(json.dumps(decision, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
