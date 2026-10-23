# Direction A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Execute inline; one independent final review after code and real smoke checks. The user explicitly delegated concrete design/execution choices, so do not request another design approval.

**Goal:** Test whether locally gated P-based correction can improve unseen-family predictions beyond fixed mixing while reducing harmful edits.

**Architecture:** A fresh frozen-embedding PairHead supplies shared base scores. Family-level cross-fitting creates correction inputs with no base exposure to the RNA's family; a group-isolated calibration set selects correctors. Slurm runs staged pilot and conditionally core experiments under allocation caps.

**Tech Stack:** Existing Python3.11/PyTorch2.1.2/ViennaRNA2.6.4/Slurm, stdlib tests and existing official metric reference.

**Spec:** docs/DIRECTION_A_PROTOCOL.md

## Global Constraints

- Historical 917dbaf experiment and numeric identity unchanged; separate checkout/output namespace.
- Pilot8 GPUh, core60 GPUh, smoke0.5 GPUh; max4 inflight; seeds17 then17/23/41.
- Fresh B0 bases50 epochs; correction15 pilot/30core; P only, frozen encoder.
- Entire outer test families absent from fitting and calibration; duplicate sequences never split correction fit/calibration.
- Full CT truth, independent metric verification, sealed identities, all failures retained.

## Review Focus

- Leakage through family-block/duplicate grouping and calibration cache provenance.
- Silent training of supposedly frozen bases and loss terms on masked positions.
- Endpoint entropy for unpaired/zero P and symmetry of local gates.
- Resume corruption, concurrent writers, source changes and allocation-cap overruns.
- Threshold/epoch selection using development or outer-test scores, incomplete output presented as success.

## Tasks

- [x] 1. Protocol/data roles: tests first for deterministic family OOF exclusion, duplicate isolation and no outer-test members. Implement scripts/direction_a_protocol.py and data adapters; serialize real pilot roles and provenance.
- [x] 2. Correction model: tests first for frozen base scores, exact convex bounds, symmetry, masked gradients and harm-loss direction. Implement scripts/direction_a_models.py. Verify CPU and actual CUDA BF16 finite gradients on real longest-RNA inputs.
- [ ] 3. Base/correction runners: implement immutable/resumable50-epoch base fitting, OOF logit caches and calibration-only correction selection. Real tiny smoke must verify coverage, optimizer steps and source binding; full input hash checks.
- [ ] 4. Baselines/evaluation: P-only, tuned mixtures, physical MFE/MEA; equal-family exact calibration; raw predictions and edits. Seal every variant before development evaluation. Independent official-function parity tests.
- [ ] 5. Bounded Slurm orchestration: smoke then pilot six shared base jobs and correction/evaluation; actual allocation ledger and promotion gates. No full core launch before measured pilot result. Source publication before formal jobs.
- [ ] 6. On promotion execute the locked core nine-fold3seed study; otherwise bounded revision or negative report. Publish tables/figures/learning and edit diagnostics, independent audit, source and report GitHub sync. Do not imply paper readiness before external/strong-baseline evidence.

## Ledger

- 2026-10-23: historical project clean at917dbaf; cluster idle for this user,26TB free. Created local Git worktree codex/direction-a and server shared-object clone RNA_ThermoFusion_direction_a. Reuse1.7GB generic embeddings and audited P caches by symlinks; no large copy/download.
- Ruling: use new B0 bases instead of historical E0 checkpoints, because those bases have trained on proposed pseudo-OOD families. Costs six small pilot bases; prevents family leakage and adds missing ArchiveII B0 control.
- Ruling: first core isolates P/gating, not Q/LoRA; priority is interpretable incremental evidence under a bounded budget.
- Task 1: complete. Three role tests observed RED then GREEN; real pilot roles sealed. SRP fit/calibration/development=1196/302/826; tmRNA=1529/378/417. Historical input identity matches the sealed study.
- Task 2: complete. Four model tests RED then GREEN. Slurm9335: 512nt real CUDA BF16 gradients finite for all four methods; no base-score gradient;2-epoch8-RNA base smoke and4 cache predictions complete. Allocation21s, smoke cap1800s.
- Task 3: base portion verified. Published source40e33ff0030f9542764b984e92c5f4c5c76a61e9 on codex/direction-a, local/server tree62c572c8189df5a9d9c1b480313c30e925259725. Four50-epoch pilot bases9336–9339 COMPLETED, allocations523/334/806/623s (2286s total). tmRNA b/full jobs9340/9341 submitted subsequently; all IDs retained in outputs/direction_a_v1/jobs.json.
- Task 4: initial metrics/edits tests RED then GREEN; nine-test suite PASS. Decoder and exact/flexible PRF match unmodified official functions on tied/empty test cases. Final real predictions still require independent verification.
- Ruling: MFE and MEA use fixed physical settings (Turner2004,37C,gamma=1 for MEA), not fictitious threshold calibration. Only thresholded P/base/mixtures/learned corrections are calibrated. Cost if wrong: physical baseline strength may depend on gamma; a later gamma sweep must be labeled separately.
- Ruling: retain the existing published plan ledger when skills upgrade from6.4.2 to7.0.1 mid-run. Do not regenerate completed tasks or rerun healthy trajectories for bookkeeping; new changes follow test-first and one final fresh review. Cost if wrong: legacy ledger has a different layout from the new helper scripts.
- Task 3: correction resume/earliest-selection tests RED→GREEN; real Slurm9342 four-method2-epoch fit/choice sealing complete(10s);9343 separate evaluation4 RNAs complete(11s), including actual ViennaRNA MFE/MEA and official metric verification. Smoke figures are never scientific result rows.
- Task 5: budget/promotion tests RED→GREEN. Real9343 submission exposed sacct visibility lag after a successful receipt; receipt now reserves full walltime without immediately re-querying sacct. Subsequent admissions still fail closed on missing accounting. Regression test RED→GREEN. Whole unit suite initially exposed missing ignored tmp directory required by an inherited test; directory created,41/41 PASS before final controller fix; full suite repeated after fix.
- Task 3: full unit suite42/42 PASS. Six base trajectories complete at50 epochs; last9341 allocation981s. Pilot fit9344 running;9345 failed46s because full base9341 was still running at submission. Keep the failure and charge it; retry only after actual summary exists (readiness verified), no base rerun.
- Final review: one fresh-context read-only reviewer on917dbaf..2eeddde. No model, split, loss or selection correctness finding. Important finding: stale promote=true could admit core after artifacts/config changed. Added validation/recomputation of current config, scientific source, roles, sealed choices, predictions and summaries; regression test RED→GREEN. Five budget tests PASS; full suite pending rerun after this fix.
- Final: minor (deferred): official metric module is not part of the training-stage source identity. Do not invalidate healthy fits for a provenance-only change. The planned independent report verifier binds that source to the historical Git object SHA256 and recomputes every final prediction; future evaluation-stage identities should include the metric module directly.
- Ruling: label later nine-fold results as a design-exposed benchmark comparison. Pilot fitting/development collectively use labels from all families except5s; per-model family exclusion does not erase research-design exposure. Only5s remains wholly reserved at this stage. Stronger paper claims need external data or outer-specific nested selection. Cost if wrong: less expansive claims, additional validation before submission; prevents an exaggerated blind-generalization claim.
