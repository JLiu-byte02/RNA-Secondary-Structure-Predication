# Direction A bounded revision1 — structured-risk calibration

Prospective revision recorded after the completed v1 development results, before any revision prediction is scored. This is the sole permitted bounded revision under the original8 GPU-hour pilot cap. It is a development refinement, not a new blind test.

## Trigger and hypothesis

v1 local gating changes SRP exact F1 by+0.019224 versus fixed mixing, but tmRNA by-0.028940. The protected variant's two-case mean gain is only+0.001824 and its mean harmful-edit fraction exceeds the residual comparator. Neither promotes. Candidate-level absolute probability preservation is not the same objective as preserving correct decoded structure. This mismatch is a hypothesis, not an established causal explanation.

## Minimal change and matched controls

Reuse all six bases, all family-OOF caches, selected v1 checkpoints and the exact fit/calibration/development IDs. No encoder, base or corrector is retrained. Revision policies scale the selected correction by beta=[0,.25,.5,.75,1]: S_beta=S0+beta*(S_corrected-S0). The three policies are fixed physical mixing, global gating and local gating. Fixed mixing has S_corrected=logit(P), exactly the same five physical weights as the prior mixture grid. Global/local reuse their own frozen selected checkpoints. There is one new calibration rule, not a search over new networks or losses.

For each case, obtain the v1 fixed-mixture and residual predictions on calibration with their already selected thresholds. Set the structured risk budget to the smaller of their equal-family mean per-RNA harmful-edit fractions relative to the same calibrated base. For each policy and every beta/threshold pair, decode on calibration and measure exact F1 plus actual harmful edits. Select the feasible policy with maximal equal-family exact F1. Ties: lower harmful fraction, smaller beta, lower threshold. Beta0 with the base threshold always supplies a feasible identity fallback. This is empirical calibration under distribution shift, not a risk guarantee.

All three policies for both cases must be sealed before any revision development evaluation. Report original policies alongside revision policies so gains due merely to a changed calibration rule can be separated from locality. Primary comparison is risk-local versus risk-fixed; report old fixed mixture as a second comparator and risk-global as the locality control.

## Budget and decision

GPU only caches calibration/development model scores; CPU handles the official reference audit. Allocate at most30min per case for calibration including inference and15min per case for revision evaluation; share caches. Charge all jobs to the existing pilot ledger, max4 inflight. No new downloads, embeddings or base fits.

Promotion still requires risk-local mean exact gain>=.005 over risk-fixed, no case worse by>.01, and mean harmful-edit fraction no higher than the v1 residual comparator. Also require nonnegative mean gain over the original fixed mixture. A failure ends this pilot with an explicit negative/limited result and no full nine-fold launch. A positive outcome warrants a newly locked confirmation plan, not an immediate claim of novelty or paper readiness.

## Integrity and publication

Original v1 scientific files remain byte-for-byte unchanged. New revision files bind the old source identity, selected checkpoint/seal hashes, roles, this prospective config and their own code hashes. The official metric source is independently bound by its historical Git-derived SHA256. If GitHub transport is unavailable, seal an immutable local Git tree/commit and verified server source bundle before revision execution, then publish the identical tree when connectivity recovers. Record the outage instead of rerunning healthy science or claiming that unpublished code has been pushed.
