# Direction A: manuscript evidence plan

This is a research plan, not a claim of a publishable result. The completed v1 and R1 pilots both failed their prospective promotion rules; the conditional nine-fold core study was not launched. See [audited stage findings](../reports/direction_a_revision1_2026-10-23/RESEARCH_FINDINGS.md) and [next model design](../reports/direction_a_revision1_2026-10-23/NEXT_MODEL_DESIGN.md). The previous E experiment is historical motivation, not a matched control for the new training protocol.

## Claim to test

A local sequence-conditioned gate can retain useful language-model structure predictions while selectively admitting thermodynamic evidence on an unseen RNA family, beyond calibrated fixed mixing and a global gate. Family-out-of-fold base predictions expose the corrector to base errors on families its base has not fitted. Harm regularization is only a surrogate; actual structural edits determine whether correction is safer.

## Evidence priority

1. Core causal comparison: same fresh B0 head, frozen representations, family blocks, calibration IDs, seeds, decoder, and thresholds grid. Compare base, P-only, fixed mix, residual CNN, global gate, local gate, and local gate with the protection term. Fixed MFE and MEA are physical references; their noncrossing constraint differs from the learned decoder.
2. Pilot mechanism: report both pseudo-OOD families and every attempted variant, exact/flexible PRF, helpful/harmful additions/removals, gate variation, and own-threshold versus common-base-threshold effects. Fit/calibration curves reveal selection and collapse. A well-calibrated identity mapping is not evidence of useful correction.
3. Conditional nine-fold confirmation: only after the prospective promotion rule, lock one candidate. Three seeds and paired per-family differences; report both equal-family and RNA-weighted results. Include worst-family changes and physical-computation, embedding, fitting, calibration, and inference costs separately. Do not use three seeds as a significance claim.
4. Publication-critical extensions after positive core evidence: a second strong frozen head/backbone and an externally sourced structure dataset with sequence/homology and pretraining-overlap disclosures. Add a controlled family-cross-fit versus in-family correction-input ablation. Reuse base caches whenever the comparison permits it. These extensions require their own bounded plan; they are not silently counted as completed by the pilot.

## Prior-art boundary

The pilot's fit/development roles collectively use labels from eight ArchiveII families. Only5s is entirely excluded during current method development. If its chosen architecture is later run on nine folds, call this a design-exposed benchmark comparison, not nine independent blind confirmations. A stronger generalization claim needs the wholly reserved5s family and a new external set, or outer-specific nested selection. This exposure is separate from the correct per-model rule that the family being predicted is absent from that model's supervised fit.

DL plus thermodynamics and expert routing are established ideas (MXfold2, MoEFold2D, BPfold and DEPfold). A sigmoid gate is not a sufficient novelty claim. A defensible contribution would combine an effective local error-correction mechanism, a controlled family-shift training study, and demonstrated transport across heads or datasets, together with reduced damaging edits at comparable predictive quality. Related-work claims must be checked against the full primary papers before submission.

## Planned manuscript structure

- Introduction: explain why an improved benchmark mean can conceal damaging corrections and family-specific failures; use the historical E1/E2 comparison as motivation with its exact limitations.
- Methods: frozen base, family cross-fitting, sequence-group meta split, seven correction inputs, bounded mixing, protection surrogate, fixed decoder, choice sealing and provenance. Explicitly disclose the base-size shift and the dependence of meta-calibration inputs on opposite-block supervised fits.
- Results: strongest simple baseline first, then local versus global routing, then edit mechanism. Include all failed pilot attempts and any prospectively bounded revision in a development appendix. Keep development and confirmation tables separate.
- Limitations: ArchiveII has already informed research design; Family-OOD describes supervised fits, not guaranteed family exclusion from language-model pretraining. Calibration under shift has no arbitrary OOD risk guarantee. Existing official RiNALMo checkpoints use another adaptation protocol. Short-RNA filtering, canonical/min-loop constraints and pseudoknot handling affect the scope.
- Reproducibility: exact source, inputs, sequence membership, seeds, selected epochs/thresholds, all allocated jobs including failures, raw predictions, independent official metrics and exportable source data.

## Stop and revise

Failure of the pilot promotion gate stops the automatic nine-fold expansion. Diagnose whether fixed physics already accounts for gains, whether gates collapse to a scalar, whether BCE suppresses sparse true pairs, and whether OOF-to-full base calibration shifts explain failure. At most one explicitly recorded revision fits within the existing8 GPU-hour pilot cap. If neither mechanism nor predictive benefit is supported, preserve a negative report and revise the research question instead of writing a positive paper narrative.
