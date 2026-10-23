# Direction A stage 2: matched calibration and structural utility

Decision by research agent under the user's explicit delegation, 2026-10-23. Reuse the completed negative v1/R1 findings, six bases, audited inputs and matching-safe edit-packet implementation. Do not rerun these fits. Keep their scientific identities and original promotion=false unchanged.

## Approaches and decision

Broader CNN/LoRA training is deferred: it would obscure the decoder/objective mismatch. First finish matched expanded calibration of every frozen candidate. Then test a small structure-edit utility selector, justified by the observed packet-vs-whole-RNA diagnostic headroom, not by treating a truth-using oracle as a model result.

## Matched comparison

Use the same thresholds (old grid plus .0005,.001,.002,.005,.01,.02,.03,.04) and strengths 0,.25,.5,.75,1 for fixed mixing and each of the four existing selected neural outputs. Epoch/checkpoint selection remains the original v1 calibration selection; this is a frozen-checkpoint, post-hoc calibration comparison, not full retraining under a new grid. Retain base and P-only controls. Select every case/method before development inference/scoring. Report unconstrained F1 choice and an empirical-harm-constrained choice using the matched fixed-mix calibration harm budget. A single expanded-calibrated base prediction is the reference for every method's edits. Raw source, parent checkpoint, cached logits, roles and seals must be bound by hashes.

Maximum additional GPU allocation: 1 hour, including failures. Cache inference at most once per case/role, then CPU calibration. CPU jobs at most four cores and 60 minutes each; maximum four active own jobs. No new full nine-fold or core admission.

## Structural utility prototype

Fresh protocol/source namespace. For each development case SRP/tmRNA exclude D and 5S from all upstream training and utility fitting. Fix the existing base_b trained on three families A. Split remaining four block_a families deterministically into two proposal-training B and two utility C families. A/B/C/D family sets must be disjoint; equal sequences must not cross roles. Fit a P-based proposal for exactly 15 epochs on B, no C-dependent checkpoint choice. Reuse the very same base/proposal at C-fit, C-cal and D; do not substitute full bases at development.

Split C by sequence groups into utility fit/cal. Compare identity, all edits, calibrated fixed mixing, margin/stem rules and a small utility MLP on atomic or stem-merged packets. Train-only labels are true change in TP and FP, never inputs. Consider selecting packets with predicted positive utility and penalties from a small calibration-only grid. Every subset must remain a valid one-partner matching, crossing allowed. No GT oracle value enters selection or promotion. Parameter-matched controls and independent official metrics are required before scientific success claims.

Additional mechanism GPU cap 2 hours, including smoke/failures; seed17 only, two cases. Admission reserves full requested walltime, not expected time. On failure or a non-promoting result, stop expansion and keep the negative findings. A positive pilot is not submission readiness: it needs seed17/23/41, a fresh external cohort and another backbone after method freeze.

Prototype gate relative to the strongest calibrated, matched simple control: mean exact F1 improvement >=.005, no case loss greater than .01 and mean harmful-edit fraction no higher. These are compute gates, not significance or arbitrary-OOD safety guarantees. ArchiveII is design-exposed; only5S remains excluded in this development stage, and historical ArchiveII results are known.
