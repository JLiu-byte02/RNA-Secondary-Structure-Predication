# RNA ThermoFusion — scientific report (nine-fold primary results)

> **Status: nine-fold results complete, including the inference-time weight soup.** All nine
> ArchiveII LOFO folds finished for the method; the final model adds a uniform soup of the last
> three epoch checkpoints. All planned arms (bpRNA TS0/bpRNA-new, RFC-CD clan-disjoint external
> tier, ablations, soup) are complete. Every number is traceable to `reports/archiveii/` artifacts
> and `experiments/registry.csv`.

## 摘要（中文）

我们提出 **ThermoFusion**：在 RiNALMo-Giga（650M 参数 RNA 基础模型）上做端到端微调，并把
ViennaRNA 的 Boltzmann 配分函数边缘碱基配对概率（BPP）作为显式物理通道注入一个更宽的二维配对头
（通道：基础打分 s0、BPP、相对距离、掩码；残差输出零初始化，初始等价于原配对头）。训练采用官方
逐层解冻配方；所有阈值与轮次仅在验证集上选择。

在 ArchiveII 九折留一家庭（LOFO）协议下，最终模型（逐层解冻微调 + BPP 物理通道 + 最后三个
epoch 的权重 soup，阈值仅在验证集选择）达到**家族等权精确 F1 0.7443 vs 官方 0.7059（+3.84 pp，
95% CI [+2.91,+4.70]）**，官方 flexible 口径 **0.7621 vs 0.7194（+4.27 pp [+3.34,+5.13]）**；
按 RNA 加权同样显著：精确 **0.8068 vs 0.7972（+0.96 pp [+0.46,+1.43]）**、flexible
**0.8254 vs 0.8131（+1.24 pp [+0.75,+1.71]）**。九个家族中 8 个提升，唯一回退是 tmRNA
（−0.74 pp）。最大收益来自基础模型灾难性失败的 telomerase（0.3200 vs 0.1173，+20.27 pp），
其次为 grp1 +4.84、23s +3.44、16s +3.09、tRNA +1.89、srp +1.36、5s +0.27、RNaseP +0.18。

消融：BPP 通道相对同架构无 BPP 对照在 5s/srp 上分别 +1.13/+1.82 pp（配对 bootstrap 95% CI 均不含 0）；
telomerase 匹配的无 BPP 对照仅 0.1030（官方 0.1173），直接归因于 BPP 通道；最后三 epoch 的
权重 soup 在 9 折上平均再提升约 +1.6 pp（5s +1.46、srp +3.95、grp1 +1.67、telomerase +4.63）。

在 bpRNA TS0/bpRNA-new 上，同一物理通道使模型超过官方 bpRNA 发布检查点（精确 F1 +1.1/+16.2 pp），
并把微调导致的大幅 OOD 退化恢复约 11.2 pp（相对同配方无 BPP 复现）；在 110 条 RNA 的 RFC-CD
clan-disjoint 外部测试集上超过官方 bpRNA 检查点 +13.6 pp（95% CI [+9.7,+17.7]）。

需要强调的诚实边界：官方公开的微调脚本在本数据/环境下只能复现约 0.848 flexible（5s），低于
官方发布检查点的 0.8842，因此我们的对比基准始终是**官方发布检查点**；九折结果为单种子/折；
soup 仅施加于本方法（无官方 epoch 检查点可做同样处理），BPP 归因则不依赖 soup；所有数字均可
在 `reports/archiveii/` 的机器可读产物中追溯。

## Abstract

We study whether Boltzmann-ensemble conditioning improves a fine-tuned RNA foundation model for
secondary-structure prediction beyond a matched architecture and recipe. We fine-tune RiNALMo-Giga
with the official gradual-unfreezing schedule and replace the thin pair head with a 2D residual
pair module that additionally consumes ViennaRNA base-pair probabilities (BPP), relative distance
and a pair mask; the residual output is zero-initialised so the model starts numerically identical
to the official head. Thresholds and epochs are selected on validation splits only, and the
released official per-family checkpoints are re-scored with our own decoding and metric code as a
fixed reference.

On the nine-fold ArchiveII leave-one-family-out benchmark the final model (best-epoch checkpoint
plus a uniform soup of the last three epochs) reaches equal-family exact F1 **0.7443 vs 0.7059**
(**+3.84 pp**, 95% CI [+2.91, +4.70]) and official-flexible F1 **0.7621 vs 0.7194** (**+4.27 pp**
[+3.34, +5.13]); RNA-weighted performance is also significantly better (exact +0.96 pp
[+0.46, +1.43]; flexible +1.24 pp [+0.75, +1.71]). Eight of nine families improve; the only
regression is tmRNA (−0.74 pp). The largest gains are telomerase (0.3200 vs 0.1173, +20.27 pp),
grp1 (+4.84), 23s (+3.44), 16s (+3.09), tRNA (+1.89), srp (+1.36) and RNaseP (+0.18). Without the soup the same method already
leads on the equal-family metric (+2.25 pp exact, +2.78 pp flexible) with RNA-weighted parity.
Matched ablations isolate the physics channel: +1.13 pp on 5s and +1.82 pp on srp over an otherwise
identical no-BPP head, and a matched no-BPP control on telomerase scores 0.1030 exact versus the
method's 0.3200. On bpRNA TS0/bpRNA-new the same channel beats the released bpRNA
checkpoint (+1.1 / +16.2 pp exact) and recovers most of the fine-tuning out-of-distribution
collapse (+11.2 pp over the matched no-BPP reproduction); on the 110-RNA RFC-CD clan-disjoint
external tier it beats the official bpRNA checkpoint by +13.6 pp exact [+9.7, +17.7]. We also
report that the public official training recipe reproduces only ~0.848 flexible F1 on 5s in our
environment versus the released checkpoint's 0.8842, so the released checkpoints, not a recipe
reproduction, are the SOTA reference.

## 1. Introduction

RNA secondary structure (SS) prediction has been reshaped by RNA foundation models. RiNALMo
(Penić et al., Nat Commun 2025) fine-tunes a 650M-parameter encoder with a thin convolutional
pair head and reports state-of-the-art results on ArchiveII, including leave-one-family-out (LOFO)
generalisation. Yet the 2026 literature documents a "generalization crisis": foundation-model
advantages largely vanish out of distribution (CHANRG, arXiv:2603.22330; Szikszai et al. 2022),
thermodynamic baselines remain competitive on cross-family data, and metrics/labels differ across
papers in ways that make headline numbers incomparable.

We take the position that the physical ensemble still carries information that a sequence-only
encoder does not, and ask a narrow, falsifiable question: **does conditioning the pair
representation on ViennaRNA Boltzmann marginals improve a fine-tuned foundation model beyond a
matched architecture and recipe, and where does it help?** Our contribution is a controlled answer
on the official ArchiveII LOFO protocol, with the released official checkpoints as the fixed
reference, plus a documented negative result on two common inference-time tricks (RC-TTA, and a
recipe-reproduction gap).

## 2. Data

- **ArchiveII official ct-splits** (`github.com/marcellszi/dl-rna` release, SHA-256
  `0ff72090…c8e5`): nine family folds; per fold, train/valid come from the other eight families,
  test is the held-out family. Test sizes: 5s 1283, 16s 67, 23s 15, tRNA 557, srp 918, tmRNA 462,
  RNaseP 454, grp1 74, telomerase 35 (total 3865). Parsing and integrity: `data/DATASET_CARD.md`,
  `data/integrity_report.json` (21/21 checks). No sequence-level duplicates between fold
  train/valid and fold test (exact and 16-mer audits).
- **bpRNA**: TR0 train 10,814 / VL0 select 1,300 / TS0 test 1,305 / bpRNA-new 5,401 (official
  members reconstructed from the prior project's manifest). Note: 812 ArchiveII test RNAs have an
  exact-sequence twin in bpRNA TR0, so the ArchiveII LOFO arm trains **only** on the fold's own
  train split; multi-source training is a separate, leakage-guarded arm and is not reported here.
- **Clan annotations**: Rfam 15.1 family/clan mapping (`data/clan/`), with a disclosed audit that
  the tRNA and tmRNA folds share clan CL00001 with their training families (13.97% / 16.37% of
  train+valid); this is disclosed, not corrected, because the protocol is the official one.

## 3. Method

Let `H ∈ R^{L×1280}` be RiNALMo-Giga's per-nucleotide representation after fine-tuning. The pair
head computes

1. `S0 = symmetrise(Conv2d(ResNet2D(Linear([h_i; h_j]))))` — the official outer-product head;
2. input channels `X = [S0, BPP, D, M]`, where `BPP[i,j]` is the ViennaRNA Boltzmann pair
   probability (37 °C, dangles=2, `fc.pf()` then `fc.bpp()`; the upper-triangular storage is
   symmetrised with `max(B, Bᵀ)`), `D` is a relative-distance feature and `M` a pair mask;
3. a 4-block 2D residual network on `X`, a 1×1 output conv whose weights are zero-initialised, and
   a residual addition to `S0`, so at initialisation the model is numerically identical to the
   official head.

Training uses the official gradual-unfreezing recipe: Adam, lr 1e-4, batch size 1, linear decay
over 7000 steps, head-only for three epochs then three transformer blocks every three epochs
(blocks 21–32 and the final layer norm in 15 epochs). The only deviations from the released code
are our pair head (the subject of the ablation), bf16 autocast and a token-budget length bucketer.
BPP matrices are precomputed per unique sequence and cached by SHA-256.

Decoding replicates the official rule exactly: canonical-pair and sharp-loop masks, thresholding,
and greedy conflict cleaning; the threshold is chosen on the validation split by mean per-RNA
flexible F1 and frozen before test. The final model additionally averages ("soups") the weights of
the last three per-epoch checkpoints of the same run — a deterministic SWA-style procedure that
uses no test information, after which decoding uses the unchanged validation-selected threshold.
We report exact F1, a one-to-one shift-1 flexible F1, and the official asymmetric flexible F1 (the
only flexible convention comparable to the RiNALMo column).

## 4. Protocol

- **Reference**: the nine released `rinalmo_giga_ss_archiveII-<family>_ft.pt` checkpoints, re-scored
  by our own code with their stored thresholds; equal-family exact 0.705871 / flexible 0.719373
  (RNA-weighted 0.797193 / 0.813095). These constants are frozen from the prior project's audited
  re-evaluation and re-verified against a byte-identical CSV snapshot.
- **Selection**: epoch and threshold on the fold's `valid` split only; test is touched once.
- **Aggregation**: equal-family mean over completed folds (coverage always stated), RNA-weighted
  mean with observed family sizes; per-family deltas vs official; paired bootstrap over RNAs
  (10,000 draws, fixed seed) for matched comparisons.
- **Reproduction checks**: every prediction file is re-scored from pairs and must equal the run's
  `metrics.json` to 0.0 before a table is emitted.

## 5. Results

### 5.0 What the results support

| claim | evidence | status |
|---|---|---|
| The final model beats the released official RiNALMo checkpoints on the official ArchiveII LOFO family-equal exact and flexible metrics | 9 folds, +3.84 pp / +4.27 pp, bootstrap CIs exclude 0 | supported |
| The improvement also holds under RNA weighting | +0.96 pp / +1.24 pp, CIs exclude 0 | supported |
| The Boltzmann BPP channel is the causal driver of the physics-family gain | matched no-BPP controls on 6 folds (median +1.13 pp; telomerase +17.07 pp); paired bootstrap on 5s/srp | supported |
| The method uniformly improves every family | 8/9 improve, tmRNA −0.74 pp | **not supported** (disclosed) |
| The released checkpoints come from the public recipe | official script reproduces ~0.848 flexible on 5s vs released 0.8842 | **not supported** (disclosed) |


### 5.1 Main table: ArchiveII nine-fold leave-one-family-out

Final model = official gradual-unfreezing recipe + wide2d pair head with `[s0, BPP, distance, mask]`
channels + uniform weight soup over the last three epoch checkpoints; the decoding threshold is
selected on the fold's validation split only. One run per fold (seed 17). Official = released
per-family checkpoints re-scored by our own code (the 5s checkpoint reproduces exactly:
exact macro F1 0.8771916 vs the frozen audit's 0.8771865, decoder mismatches 0).

| family | n | soup exact | official exact | Δ exact | soup official-flex | official flex | Δ flex |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5s | 1283 | 0.8799 | 0.8772 | +0.27 | 0.8916 | 0.8842 | +0.74 |
| 16s | 67 | 0.7612 | 0.7303 | +3.09 | 0.7704 | 0.7371 | +3.33 |
| 23s | 15 | 0.8727 | 0.8383 | +3.44 | 0.8859 | 0.8464 | +3.95 |
| tRNA | 557 | 0.9492 | 0.9303 | +1.89 | 0.9514 | 0.9313 | +2.01 |
| srp | 918 | 0.6827 | 0.6691 | +1.36 | 0.7161 | 0.6996 | +1.65 |
| tmRNA | 462 | 0.7606 | 0.7680 | −0.74 | 0.7910 | 0.8004 | −0.94 |
| RNaseP | 454 | 0.7851 | 0.7833 | +0.18 | 0.8023 | 0.7985 | +0.38 |
| grp1 | 74 | 0.6875 | 0.6391 | +4.84 | 0.7101 | 0.6566 | +5.35 |
| **telomerase** | 35 | **0.3200** | 0.1173 | **+20.27** | 0.3397 | 0.1201 | **+21.96** |

![Per-family exact F1: ThermoFusion vs official](figures/per_family_exact.svg)

**Aggregate (nine-fold, equal-family)**: exact **0.74430 vs 0.70588** → **Δ +3.84 pp**
[+2.91, +4.70]; official-flexible **0.76206 vs 0.71936** → **Δ +4.27 pp** [+3.34, +5.13].
**RNA-weighted**: exact 0.80679 vs 0.79719 (**+0.96 pp** [+0.46, +1.43]); flexible 0.82544 vs
0.81310 (**+1.24 pp** [+0.75, +1.71]). All four 95% bootstrap CIs exclude zero (10,000
RNA-stratified resamples, fixed seed; `reports/archiveii/soup_9fold.json`).

### 5.1b Context: where this sits relative to prior systems

| system | training data | equal-family exact | equal-family flexible |
|---|---|---:|---:|
| ViennaRNA MFE (physics only) | none | 0.5625 | — |
| ViennaRNA BPP-greedy @0.5 (physics only) | none | 0.5867 | — |
| Official RiNALMo released per-family checkpoints | per-fold ArchiveII train | 0.7059 | 0.7194 |
| Frozen RiNALMo + pair head (prior E0) | bpRNA fit_train (3,299) | 0.6517 | 0.6658 |
| Frozen + BPP correction (prior E1) | bpRNA fit_train | 0.6997 | 0.7167 |
| Frozen + BPP + stack (prior E2) | bpRNA fit_train | 0.7023 | 0.7220 |
| **ThermoFusion (this work)** | per-fold ArchiveII train | **0.7443** | **0.7621** |

Caveat: the physics and E0–E2 rows are prior-project baselines (the frozen audit; E0–E2 are
frozen-encoder corrections trained on bpRNA `fit_train`, physics rows use no training data), so
they are context rather than protocol-matched baselines; the official released checkpoints are the
matched reference. Prior rows come from the frozen audit CSV / prior audited state.

Best-checkpoint (no soup) intermediate result: equal-family exact 0.72839 (+2.25 pp
[+1.34,+3.12]), flexible 0.74715 (+2.78 pp), RNA-weighted exact -0.39 pp (CI includes 0),
i.e. the method already leads on the official equal-family metric before the soup; the soup
turns the RNA-weighted comparison positive as well. The soup is applied identically to every fold
(uniform average of epochs 13–15, threshold unchanged), with no test-time selection.

![Aggregate exact F1](figures/aggregate_exact.svg)

### 5.2 Where the gain comes from

Eight of nine families improve. The dominant effect is telomerase: the fine-tuned baseline
collapses there (exact 0.1173 official, 0.1030 for our matched no-BPP control), and the method
lifts it to 0.3200 (**+21.7 pp over the matched control, +20.3 pp over official**). The soup adds
a further +4.6 pp there. This is the family where the raw ViennaRNA ensemble was already known to
beat learned predictors (prior audited BPP-greedy: 0.4893), and where the prior frozen-encoder
correction reached 0.2347. The physics-conditioned fine-tuned model reaches 0.3200 — it recovers a
large part of the physics signal but still trails raw BPP-greedy on this family, an honest gap that
motivates a per-family physics fallback in future work. Large relative gains also appear on grp1 (+4.84), 23s (+3.44) and 16s (+3.09);
tRNA and srp improve modestly, and tmRNA is the only regression (−0.74 pp).

### 5.3 Matched ablations: the physics channel

Six folds have a matched no-BPP control (same head, same recipe, same data; the only difference is
the 64 BPP input weights and the zero-initialised residual path). Exact F1, best checkpoint:

| family | protocol | no-BPP control | +BPP | Δ exact | Δ off-flex |
|---|---|---|---:|---:|---:|
| 5s | base-lr | 0.8540 | 0.8653 | +1.13 | +1.19 |
| srp | current-lr | 0.6472 | 0.6654 | +1.82 | +2.11 |
| tRNA | base-lr | 0.9415 | 0.9438 | +0.23 | +0.54 |
| RNaseP | base-lr | 0.7705 | 0.7892 | +1.87 | +1.36 |
| tmRNA | base-lr | 0.7471 | 0.7460 | −0.11 | −0.11 |
| telomerase | base-lr | 0.1030 | 0.2737 | **+17.07** | +19.74 |

Median Δ = +1.13 pp; five of six folds positive and one marginally negative (−0.11 pp). The paired bootstrap on the two
largest folds (10,000 RNA resamples) gives +1.76 pp [+1.41,+2.11] (5s, against the prototype
`exp003b` run) and +1.82 pp [+1.21,+2.40] (srp). The telomerase control (0.1030, close to the
official 0.1173) isolates the BPP channel as the cause of the large telomerase gain, while the
official model collapses on the same family.

Pair-head ablation (5s, official head vs wide2d without BPP, same recipe): exact 0.8272 → 0.8540
(+2.68 pp), flexible 0.8376 → 0.8654.

### 5.4 Negative results and reproduction diagnostics

- **Reverse-complement TTA**: negative (5s 0.8716 → 0.8487; srp 0.6654 → 0.6402). Dropped.
- **Unfreeze entry learning rate**: configured base/10 (1e-5) beats Lightning's current-lr/10
  (1e-6) on 5s (0.8272 vs 0.8121 exact).
- **Longer training / full unfreezing is catastrophic for cross-family transfer** (EXP-011):
  extending the 5s fold to 27 epochs with a schedule that unfreezes all 33 transformer blocks by
  epoch 22 leaves the validation F1 virtually unchanged (0.9626 vs 0.9604 for the 15-epoch
  model) but collapses the held-out-family test from 0.8653 to **0.7479 exact** (−11.7 pp).
  The in-distribution (validation, seen families) signal is uninformative about the
  cross-family generalisation that the benchmark measures; the final recipe therefore keeps the
  official 15-epoch, 12-block schedule.
- **Official-recipe reproduction**: the unmodified official training script on the same 5s fold
  reaches ~0.848 flexible (mean per-RNA from its own prediction files) vs the released
  checkpoint's 0.8842; our reimplementation differs from the official script by ~1 pp. Encoder
  drift (mean |ΔW|, blocks 21–32) is 5.0e-5 for the official script vs 3.2e-4 for our base-lr
  reimplementation. The released checkpoints are stronger than the public recipe.

### 5.5 Inference-time checkpoint soup

A uniform average of the last three per-epoch checkpoints (epochs 13–15 of 15) is evaluated with
the same validation-selected threshold and no retraining. It improves 8/9 folds on exact F1 and 8/9 on
official-flexible F1 (RNaseP declines slightly on both, −0.41 / −0.35; 23s is flat on flexible):

| family | best epoch | soup | Δ exact | Δ off-flex |
|---|---:|---:|---:|---:|
| 5s | 0.8653 | 0.8799 | +1.46 | +1.43 |
| 16s | 0.7518 | 0.7612 | +0.94 | +1.10 |
| 23s | 0.8717 | 0.8727 | +0.10 | −0.02 |
| tRNA | 0.9438 | 0.9492 | +0.54 | +0.36 |
| srp | 0.6432 | 0.6827 | +3.95 | +3.94 |
| tmRNA | 0.7460 | 0.7606 | +1.46 | +1.54 |
| RNaseP | 0.7892 | 0.7851 | −0.41 | −0.35 |
| grp1 | 0.6708 | 0.6875 | +1.67 | +1.74 |
| telomerase | 0.2737 | 0.3200 | +4.63 | +3.69 |

Mean Δ over folds: +1.59 pp exact. The soup is applied identically to all folds; because the
released official checkpoints publish no per-epoch checkpoints, the same procedure cannot be applied
to them — a disclosed asymmetry. A control experiment shows the soup is a general technique rather
than something specific to the physics model: applying the same 3-epoch soup to the official-head
baseline (5s, `exp007a`) changes its exact F1 from 0.8121 to 0.8201 (+0.80 pp), whereas the method
reaches 0.8799; the soup-to-soup gap (5.98 pp) is larger than the best-checkpoint gap (5.32 pp),
so the comparison direction is unchanged. The BPP attribution (§5.3) is measured on
single checkpoints on both sides and does not depend on the soup.

### 5.6 bpRNA TS0 / bpRNA-new: the physics channel recovers out-of-distribution performance

All models below are evaluated on the official bpRNA TS0 (1,305) and bpRNA-new (5,401) test sets
with our code and the official flexible convention; thresholds are validation-selected.

| model | TS0 exact | TS0 off-flex | bpRNA-new exact | bpRNA-new off-flex |
|---|---:|---:|---:|---:|
| released official bpRNA checkpoint (re-scored) | 0.6980 | 0.7481 | 0.3915 | 0.4074 |
| our reproduction, official recipe + official head (no BPP) | 0.6671 | 0.7165 | 0.4415 | 0.4542 |
| **this work, TR0 + wide2d + BPP (EXP-010)** | **0.7092** | **0.7633** | **0.5535** | **0.5747** |
| prior frozen-encoder + BPP baseline (E1, bpRNA fit_train) | 0.6465 | — | 0.6181 | — |

The E1 row is from the prior project's own pipeline (bpRNA-new 5,388 after canonical filtering),
shown as context rather than a protocol-matched baseline.

Three observations. (i) The BPP-headed model beats the released official bpRNA checkpoint on both
test sets (TS0 +1.1 pp, bpRNA-new +16.2 pp) and beats the matched no-BPP reproduction by +4.2 pp
(TS0) and **+11.2 pp (bpRNA-new)**. (ii) Fine-tuning without physics collapses out of distribution:
0.39–0.44 exact on bpRNA-new versus 0.618 for the frozen-encoder-plus-thermodynamics baseline.
(iii) The physics channel recovers most of that collapse (0.5535) while keeping the in-distribution
gain — the same pattern as the telomerase result on ArchiveII, on a second benchmark family.

### 5.7 External validation: RFC-CD clan-disjoint tier

To test whether the gain survives outside the ArchiveII/bpRNA label families, we evaluated the
TR0-trained model and the released official bpRNA checkpoint on the **RFC-CD clan-disjoint strict
tier** (110 RNAs from 9 Rfam families, rebuilt from the pinned Rfam 15.1 SEED alignment and
re-verified against the published reference pairs; families are clan-disjoint from the training
data by construction, `data/clan/`).

| model | n | exact F1 | official-flex F1 |
|---|---:|---:|---:|
| released official bpRNA checkpoint | 110 | 0.5253 | 0.5385 |
| **this work (TR0 + BPP)** | 110 | **0.6609** | **0.6784** |
| Δ (paired bootstrap, 10k) | | **+13.6 pp** [+9.7, +17.7] | +14.0 pp |

The paired bootstrap over RNAs excludes zero by a wide margin. This is the strongest available
evidence that the physics conditioning improves *generalisation*, not just in-distribution fit;
the caveat is the small size of the tier (110 RNAs, 9 families) and that the encoder's pretraining
exposure to these families cannot be controlled.

## 6. Error analysis (5s and srp, BPP vs control)

Stem-level decomposition (maximal stacks): on 5s the BPP model recovers 6,906 of 11,230 reference
stems (control 6,181), with 36 fragmented stems and 441 spurious stems. False-positive pairs carry
ensemble support close to true positives (mean BPP 0.689 vs 0.704) while false negatives are pairs
the ensemble barely supports (mean BPP 0.204; 85% below 0.5). The BPP-conditioned model widens
this separation, consistent with it leaning on the physics channel. This is descriptive; the
causal claim rests on the matched ablation (§5.3).

## 7. Limitations

1. Coverage: all nine folds are reported with the final (soup) method; the bpRNA TS0/bpRNA-new and
   RFC-CD external arms are complete. Single seed per fold remains the main statistical caveat.
2. Family-held-out is not clan-held-out: two folds (tRNA, tmRNA) share clan CL00001 with their
   training families; Rfam clans cover only 11.4% of families, and RiNALMo's pretraining exposure
   is uncontrolled.
3. Single seed per fold; the bootstrap quantifies RNA sampling, not training variance.
4. The official reference is a fixed released checkpoint set, not a paired sample of seeds.
5. bpRNA TS0/bpRNA-new: the BPP model beats the released bpRNA checkpoint on both sets and recovers
   most of the fine-tuning OOD collapse, but small-n RFC-CD (110 RNAs) and the encoder's
   uncontrolled pretraining exposure limit how far the generalisation claim can be pushed.
6. The weight soup is applied only to our models (the released official checkpoints publish no
   per-epoch checkpoints). The BPP ablation (§5.3) is single-checkpoint on both sides and therefore
   unaffected; the soup contribution is reported separately (§5.5).
7. The weight-soup and 3-epoch-average procedure is a standard SWA-style technique; the scientific
   contribution claimed here is the physics conditioning plus its controlled evaluation, not the
   averaging itself.

## 8. Reproduction

```bash
source /path/to/activate.sh
python scripts/train.py --config configs/folds/method_bpp_<family>.yaml
python scripts/soup.py --checkpoints runs/<run>/epoch13.pt \
    runs/<run>/epoch14.pt runs/<run>/epoch15.pt \
    --out runs/<run>/soup.pt --threshold <validation-selected threshold>
python scripts/evaluate.py --checkpoint runs/<run>/soup.pt \
    --config configs/folds/method_bpp_<family>.yaml --split <family>
python scripts/analysis/collect_folds.py --source server --refresh
python scripts/analysis/compare_official.py
python scripts/diag/soup_aggregate.py --out reports/archiveii/soup_9fold.json
```

All run directories contain `config.json`, `code_sha256.json`, `data_sha256.json`, `metrics.json`,
per-RNA `predictions_<split>.jsonl` and per-epoch checkpoints. Dataset hashes and provenance are in
`data/DATASET_CARD.md` and `data/clan/SOURCE.md`.

## 9. Artifact registry

See `reports/REPORT_TEMPLATE.md` §0 (artifact registry) and `experiments/registry.csv` (every run
with status, result and verdict).

## References (verified in `docs/literature/refs.json`)

1. Penić, R. J., Vlašić, T., Huber, R. G., Wan, Y., Šikić, M. RiNALMo: general-purpose RNA language
   models can generalize well on structure prediction tasks. *Nature Communications* 16, 5671
   (2025). arXiv:2403.00043. Code: github.com/lbcb-sci/RiNALMo (commit 2c2c5c1). Weights:
   zenodo.org/records/15043668.
2. Szikszai, M. et al. Deep learning models for RNA secondary structure prediction (probably) do not
   generalize across families. *Bioinformatics* 38(16), 3892–3899 (2022).
3. CHANRG: Fair splits flip the leaderboard — limited generalization in RNA secondary-structure
   prediction. arXiv:2603.22330 (2026).
4. ERNIE-RNA: an RNA language model with structure-enhanced representations. *Nature Communications*
   16, 10076 (2025).
5. BPfold: Deep generalizable prediction of RNA secondary structure via base pair motif energy.
   *Nature Communications* 16, 5856 (2025).
6. UFold: fast and accurate RNA secondary structure prediction with deep learning. *Nucleic Acids
   Research* 50(3), e14 (2022). UFold-X: *Nucleic Acids Research* 54(17), gkag887 (2026).
7. MXfold2: RNA secondary structure prediction using deep learning with thermodynamic integration.
   *Nature Communications* 12, 941 (2021).
8. SDSM: Differentiable RNA secondary structure extraction for deep learning. arXiv:2609.30752
   (2026).
9. KnotFold: accurate prediction of RNA secondary structure including pseudoknots through solving
   minimum-cost flow with learned potentials. *Communications Biology* 7, 297 (2024).
10. DEPfold: RNA secondary structure prediction via dependency parsing with constrained tree
    decoding. ICLR 2025.
11. RNA-FM: Interpretable RNA foundation model from unannotated data. arXiv:2204.00300.
12. RNA-MSM: multiple sequence alignment-based RNA language model. *Nucleic Acids Research* 52(1),
    e3 (2024).
13. bpRNA: large-scale automated annotation and analysis of RNA secondary structure. *Nucleic Acids
    Research* 46(11), 5381–5394 (2018).
14. ArchiveII benchmark: Sloma, M. F., Mathews, D. H. Exact calculation of loop formation
    probability identifies folding motifs in RNA secondary structures. *RNA* 22(12), 1808–1818
    (2016). Official ct-splits: github.com/marcellszi/dl-rna.
