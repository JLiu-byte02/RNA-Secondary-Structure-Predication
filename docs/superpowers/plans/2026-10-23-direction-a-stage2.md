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

- [ ] 1. Write failing tests for fair profile selection, common reference, role chain and packet utility; implement small pure helpers.
- [ ] 2. Implement immutable frozen-output cache and calibration/sealing/scoring commands; actual smoke with source binding; run full selected-record calibration and independent metrics under1GPUh.
- [ ] 3. Implement and test A/B/C roles, fixed-horizon proposal, utility features and simple/learned packet controls; actual GPU smoke then bounded two-case mechanism run.
- [ ] 4. Independently recompute complete predictions/edits, write actual results, resource ledger, failure mechanisms and paper readiness decision. One final fresh review of new code.
- [ ] 5. Restore GitHub branch publication (old transport now recovered), synchronize new source/results, verify native/API/server trees. Do not merge into main or claim publication before success.

## Ledger

- Ruling: local preserved worktree already includes complete v1/R1 and negative diagnostics absent from the current chat summary. Continue from these artifacts rather than duplicating runs; cost if wrong: fewer redundant confirmation runs, mitigated by hash validation.
- Ruling: existing old94-file staged snapshot stays intact until connector publication; append new stage files and publish an auditable combined update if needed, never discard pending results.
