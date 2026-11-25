# Ablation tables (auto-generated)

## TR0 (bpRNA TS0)

| configuration | status | TS0 exact F1 | TS0 official-flex F1 | τ | best epoch |
|---|---|---|---|---|---|
| Seq-Only (official 2-block head) | completed | 0.6965 | 0.7460 | 0.05 | 15 |
| Seq-Only (head-matched, no BPP) | completed | 0.6965 | 0.7466 | 0.13 | 13 |
| Seq-Only head-matched seed 1024 | completed | 0.6907 | 0.7416 | 0.12 | 11 |
| Seq-Only head-matched seed 2026 | completed | 0.6926 | 0.7456 | 0.06 | 15 |
| ThermoFusion (single best) | completed | 0.7040 | 0.7631 | 0.07 | 15 |
| ThermoFusion seed 1024 | completed | 0.7135 | 0.7693 | 0.08 | 15 |
| ThermoFusion seed 2026 | completed | 0.7106 | 0.7656 | 0.1 | 14 |
| Zero-BPP control | completed | 0.6996 | 0.7518 | 0.08 | 15 |
| Deep Seq-Only (6 blocks) | completed | 0.6951 | 0.7486 | 0.09 | 15 |
| w/o S0 | completed | 0.7145 | 0.7667 | 0.06 | 15 |
| w/o distance | completed | 0.7118 | 0.7678 | 0.1 | 14 |
| w/o mask | completed | 0.7142 | 0.7679 | 0.08 | 15 |
| BPP-only | completed | 0.7132 | 0.7651 | 0.08 | 15 |
| MFE-replaced | completed | 0.7106 | 0.7633 | 0.09 | 15 |
| multi-thermo | completed | 0.7144 | 0.7695 | 0.08 | 15 |

## Seed summary (TR0)

| configuration | n seeds | TS0 mean exact F1 | std |
|---|---|---|---|
| Seq-Only | 3 | 0.6933 | 0.0024 |
| ThermoFusion | 3 | 0.7094 | 0.0040 |

## ArchiveII fold-level core ablation

| fold | Seq-Only | ThermoFusion | Zero-BPP | Deep Seq-Only |
|---|---|---|---|---|
| 5s | 0.8547 | missing | — | — |
| srp | 0.5770 | missing | — | — |
| RNaseP | 0.7114 | missing | — | — |
| tRNA | 0.9219 | missing | — | — |
| telomerase | 0.1247 | missing | — | — |
| tmRNA | 0.6960 | missing | — | — |

## Inference ablation: tr0_thermo_s17

_inference results missing for tr0_thermo_s17_

## Inference ablation: tr0_seq_s17

_inference results missing for tr0_seq_s17_
