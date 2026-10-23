# Direction A v1 — reliable local thermodynamic correction

Status: prospective design, 2026-10-23. User authorized autonomous execution of Direction A, prioritizing core evidence and compute. Historical study 917dbaf remains read-only. No new claim of novelty or performance is made before experiments.

## Research question and model

Can a frozen base predictor be corrected with sequence-derived thermodynamic information while reducing harmful edits on unseen RNA families, beyond a tuned scalar mixture and an ungated CNN?

Train a fresh B0 PairHead on frozen generic RiNALMo embeddings; fix its weights before correction. The base training horizon is 50 epochs without validation-driven checkpoint selection. Reuse only the audited generic embeddings and P caches, not old supervised checkpoints.

Methods: base, P-only, fixed logit mixtures alpha=[0,.25,.5,.75,1], ungated residual CNN, sequence-global gate, pair-local gate, and pair-local gate with harm regularization lambda=.1. All learned correctors use the same seven channels: clipped base logit, physical logit, endpoint physical entropies, base/physical disagreement, normalized distance and valid mask. All use the same two 3x3/32-channel GELU backbone. Gates sigmoid to [0,1], start with final bias -4, and produce S=S0+g*(logit(P)-S0), bounded between the two experts. The ungated CNN starts at zero residual and is a capacity control, not an equivalent hypothesis class.

P is clipped to [1e-4,1-1e-4] for logit features only. Full CT truth is preserved for evaluation. No Q, online encoder adaptation or new dependencies in the core experiment. P entropy includes the unpaired probability; numerical normalization is checked. The correction objective is original per-RNA candidate BCE plus an optional class-balanced mean absolute probability change on correctly classified base candidates. This auxiliary term is a training surrogate, not a guarantee against structural harm or arbitrary OOD risk.

## Pilot and leakage boundary

Exclude 5s entirely from development. Use two predeclared pseudo-OOD cases, srp and tmRNA, separately. In each case D, fit on official fold-5s TRAIN IDs from the other seven families; the entire family D is excluded from every base and corrector fitting step. Its TRAIN members serve as development evaluation, never as fitting data. No outer-5s test inference during development.

Partition the seven fit families into two blocks by deterministic family-name hash, balancing family counts. Train base A on block A, base B on block B, and base full on A+B. OOF correction inputs for A come only from base B, and vice versa: the base that predicts an RNA has never trained on its family. Base full predicts D. No D labels determine base epochs or fit a corrector.

Within OOF A+B, split sequence-hash groups into correction-fit (80%) and calibration (20%), stratified by family. Identical sequences remain in one role; fail on cross-family identical sequences. Corrections train on correction-fit only. Epoch and decision threshold are selected on calibration with equal-family exact F1; earliest epoch/lower threshold ties. Different expert training sizes in OOF versus full prediction are disclosed, not assumed innocuous. Learned methods share splits, base/cache and seed. Seed17;15 correction epochs; all choices locked before D evaluation. P-only and mixtures are independently calibrated on the same calibration set. MFE/MEA use fixed Turner2004/37C settings and gamma=1 for MEA, with no development tuning. Evaluate exact/flexible P/R/F1, edit counts and common-threshold sensitivity. Calibration RNAs have family-unseen base inputs, but their labels can occur in the base used to predict the opposite block; this meta-level split is not an independent end-to-end risk-calibration sample. Entire development families remain excluded from all fitting.

Pilot promotion: a local gate must improve mean pseudo-OOD exact F1 over the best fixed mixture by >=0.005, lose <=0.01 in either D, and have no higher mean harmful-edit rate than the ungated residual comparator. This is a compute decision, not statistical significance. A failure produces a documented negative result and at most one bounded revision under the same pilot budget, not automatic full-scale training. All evaluated variants retained.

## Core confirmation

Selection exposure must be distinguished from per-model supervised exclusion. This pilot uses labels from eight ArchiveII families during fitting or development; only5s is wholly reserved at this research-design stage. A later globally selected method evaluated on all nine folds is a benchmark comparison with design exposure, not nine independent blind confirmations. Report this exposure explicitly. Clean evidence requires the reserved5s case, a newly audited external evaluation, or a prospectively nested method-selection procedure that excludes each outer family's labels. No nine-fold table alone establishes that stronger claim.

Only after promotion lock the winning architecture, lambda and objective. For each official outer family F and seeds17/23/41, train fresh cross-fitting A/B and full base models on official TRAIN IDs from the other eight families. Generate family-out-of-fold correction training inputs. Correction-fit/calibration are sequence-group disjoint within TRAIN. Official VALID IDs are a separate diagnostic role, not fitting or method-selection data in this strict protocol; their known train duplicates are disclosed. The full base predicts official TEST F, with all 27 correction choices sealed before any core test. Baselines share these exact new base models and calibration; old E tables are historical references, not identical-protocol causal controls. Core correctors train30 epochs. Full CT membership remains unchanged for outer tests.

An external RNA3DB-2D evaluation and second encoder are contingent extensions after a positive core result and a separate data-overlap audit. They do not displace the core comparison. ArchiveII is already a known benchmark, not a new blind discovery set.

## Resources and integrity

Pilot allocation cap8 GPU-hours including retries; core cap60 additional GPU-hours,4 jobs in flight at most. Track sacct allocation seconds, not epoch time alone; never cancel other users' jobs. If the cap or promotion rule prevents expansion, stop that phase with an explicit report and preserve all outputs. Smoke tests have a separate <=0.5 GPU-hour ceiling. One bounded revision may be taken only within the pilot cap. Baseline OOF computation is shared across variants, no repeated encoder downloads/embedding generation. GPU pilot walltime is measured before refining the core estimate.

Use a separate codex/direction-a checkout and outputs/direction_a_v1 namespace. Link existing feature/embedding/env directories read-only by convention; scripts must not write historical namespaces. Seal source/config/input/role/checkpoint/cache hashes; fail changed source on resume. Resume from complete atomic epochs only. Report every attempted job, failures and post-hoc observations. Final prediction metrics independently checked with unmodified official functions; no synthetic result rows.

Paper contribution is conditional: selective local correction under family shift, cross-fitted training and reduced harmful edits across tasks/backbones. A sigmoid gate, frozen encoder or better benchmark mean alone is insufficient; compare MXfold2, MoEFold2D, BPfold and modern RNA language-model heads when their data and execution are auditable. Negative results revise the research claim rather than being hidden.
