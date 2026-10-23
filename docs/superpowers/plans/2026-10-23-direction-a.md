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

- [ ] 1. Protocol/data roles: tests first for deterministic family OOF exclusion, duplicate isolation and no outer-test members. Implement scripts/direction_a_protocol.py and data adapters; serialize real pilot roles and provenance.
- [ ] 2. Correction model: tests first for frozen base scores, exact convex bounds, symmetry, masked gradients and harm-loss direction. Implement scripts/direction_a_models.py. Verify CPU and actual CUDA BF16 finite gradients on real longest-RNA inputs.
- [ ] 3. Base/correction runners: implement immutable/resumable50-epoch base fitting, OOF logit caches and calibration-only correction selection. Real tiny smoke must verify coverage, optimizer steps and source binding; full input hash checks.
- [ ] 4. Baselines/evaluation: P-only, tuned mixtures, physical MFE/MEA; equal-family exact calibration; raw predictions and edits. Seal every variant before development evaluation. Independent official-function parity tests.
- [ ] 5. Bounded Slurm orchestration: smoke then pilot six shared base jobs and correction/evaluation; actual allocation ledger and promotion gates. No full core launch before measured pilot result. Source publication before formal jobs.
- [ ] 6. On promotion execute the locked core nine-fold3seed study; otherwise bounded revision or negative report. Publish tables/figures/learning and edit diagnostics, independent audit, source and report GitHub sync. Do not imply paper readiness before external/strong-baseline evidence.

## Ledger

- 2026-10-23: historical project clean at917dbaf; cluster idle for this user,26TB free. Created local Git worktree codex/direction-a and server shared-object clone RNA_ThermoFusion_direction_a. Reuse1.7GB generic embeddings and audited P caches by symlinks; no large copy/download.
- Ruling: use new B0 bases instead of historical E0 checkpoints, because those bases have trained on proposed pseudo-OOD families. Costs six small pilot bases; prevents family leakage and adds missing ArchiveII B0 control.
- Ruling: first core isolates P/gating, not Q/LoRA; priority is interpretable incremental evidence under a bounded budget.
