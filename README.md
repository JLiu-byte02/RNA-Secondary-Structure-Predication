# RNA-Secondary-Structure-Predication

ThermoFusion predicts RNA secondary structure by combining RiNALMo-Giga sequence embeddings with thermodynamic base-pair probabilities in a residual 2D pair head.

![ThermoFusion architecture](fig1_architecture.png)

- [PyTorch model architecture](model.py)
- [Scientific report](FINAL_REPORT.md)
- [Ablation tables](ABLATION_TABLES_2026-11-05.md)
- [Completed run summary](FINAL_RUN_SUMMARY.csv)
- [Complete architecture and results archive](ThermoFusion-architecture-and-results.zip)

In the ArchiveII nine-fold leave-one-family-out evaluation, ThermoFusion reached an equal-family exact F1 of **0.74430**, compared with **0.70588** for the released RiNALMo checkpoints rescored under the same protocol. The scientific report documents the protocol and limitations. The archive preserves the original directory structure and includes the full set of selected reports, figures, and evaluation metrics.
