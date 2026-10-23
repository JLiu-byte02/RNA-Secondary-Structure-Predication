# Direction A Stage 2 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. User delegated decisions and quota allocation; no further design approval is needed. Reuse existing worktree and staged historical publication.

**Goal:** Resolve the matched-calibration confound, then test structural utility learning under bounded resources.

**Architecture:** Cache frozen v1 outputs once, choose policies on calibration with equal family weights and a common base reference, globally seal choices before development scoring. A separate A/B/C family chain tests matching-safe edit utility without exposing utility labels upstream.

**Tech Stack:** Existing Python3.11, PyTorch2.1.2, NumPy, Slurm, original official metrics.

**Spec:** docs/DIRECTION_A_STAGE2.md

## Global Constraints

- Preserve v1/R1 source and promotion=false; do not modify historical outputs.
- Matched phase cap1GPUh; mechanism cap2GPUh; <=4 own inflight jobs; include all failures/reservations.
- Do not train on 5S or development-family records; no GT input to selection; common base for edit counts.
- Cache/source/checkpoint/role hashes must match; no stale success artifacts or silent resume.

## Review Focus

- Shared threshold/reference definitions across methods and identity fallback.
- Source/checkpoint mutation after calibration but before development inference.
- C-family labels exposed to upstream model/epoch selection or full-base substitution.
- Packet subsets creating nucleotide conflicts or a truth-dependent inference path.
- Budget reservations, missing sacct records and negative pilots accidentally enabling core.

## Tasks

- [x] 1. Write failing tests for fair profile selection, common reference, role chain and packet utility; implement small pure helpers.
- [x] 2. Implement immutable frozen-output cache and calibration/sealing/scoring commands; actual smoke with source binding; run full selected-record calibration and independent metrics under1GPUh.
- [x] 3. Implement and test A/B/C roles, fixed-horizon proposal, utility features and simple/learned packet controls; actual GPU smoke then bounded two-case mechanism run.
- [x] 4. Independently recompute complete predictions/edits, write actual results, resource ledger, failure mechanisms and paper readiness decision. One final fresh review of new code.
- [ ] 5. Restore GitHub branch publication (old transport now recovered), synchronize new source/results, verify native/API/server trees. Do not merge into main or claim publication before success.

## Ledger

- Ruling: local preserved worktree already includes complete v1/R1 and negative diagnostics absent from the current chat summary. Continue from these artifacts rather than duplicating runs; cost if wrong: fewer redundant confirmation runs, mitigated by hash validation.
- Ruling: existing old94-file staged snapshot stays intact until connector publication; append new stage files and publish an auditable combined update if needed, never discard pending results.
- Source publication restored: GitHub codex/direction-a8c14d694f6d2518cf067e9db7d89531e5769f6cb, native tree9723ded357e1209410f58524be9e4c0e8942813b,111 files including prior94 pending artifacts. main remains917dbaf.
- Matched cache9364/9368 complete130/136GPU seconds; calibration9370/9374 complete44/38CPU seconds; evaluations9376/9377 complete40/47CPU seconds. No v1/R1 refit.
- One independent review found JSON pair reconstruction and missing loader/audit dependency binding. Regression tests RED→GREEN; full suite66PASS. Utility binding fixed before formal fits; matched cache-producing source stays byte-identical to46ffd snapshot with a canonical JSON-normalizing audit entry and supplemental exact dependency manifest. Audit9380 failed on JSON types46CPU seconds; retry9393 uses the canonical audit, not regenerated predictions. Existing report subcommand is frozen legacy; use slurm/direction_a_matched_audit.sbatch.
- Smoke9375 failed7GPU seconds on an empty tiny proposal; retry9379 passed8GPU seconds and explicitly does not claim utility training or performance. Formal utility9394/9395 still PENDING at inspection, reserved1200seconds each under2GPUh mechanism cap.
- Ruling: matched development logits may be cached label-free before choice sealing; only calibration labels enter selection, and all case seals precede development scoring. This wording corrects an overly strict inference-time sentence in the spec; no label/score was used to select a method. Cost if wrong: an overstated temporal protocol, mitigated by distinct label-free caches and calibration/development entry paths.
- GPU scheduler predicts multi-hour wait. CPU throughput pilot9398 is limited300seconds/4cores and noGPU. Decide hardware from measured throughput, retaining the same B0 checkpoint and recomputing B/C/D consistently if migrating toFP32CPU. Do not modify formal GPU source until own pending jobs are confirmed cancelled with zero allocation.

- Stage2 final: matched9393 and utility9417 audits both passed14916 prediction views each. Complete67 tests passed. Controller completed. All18 own jobs terminal (14completed,2failed,2cancelled);281GPU seconds and4.817778CPU core-hours including failures. CPU fits9401/9403 completed900/689seconds; both evals completed. No core admission.
- Main stem_mlp meanF1=.6906080821 versus C_cal-selected fixed_mix=.6914789978; gain=-.0008709157, per-case+.0276547055/-.0293965369. Negative gate retained; no extra seeds or ninefold launch. Next: bounded current-chain candidate/utility diagnostics before any new fit.
- Numerical source is frozen and bound to FP32CPU runtime. Two final plots retain all24rows each; text/collision/source QA and visual review completed. Final result publication and server reconciliation are verified in the task-local final attestation, separately from source commit8c14.
