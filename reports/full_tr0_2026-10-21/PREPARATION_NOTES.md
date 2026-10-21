# 全量实验准备记录

Slurm 作业 9132 已完成四个固定集合的表征生成、标签策略和划分审计、P/Q 缓存，以及 E0/E1/E2 的 smoke test。样本数为 TR0 10,682、VL0 1,286、TS0 1,288、bpRNA-new 5,388；没有跨集合 ID 或序列重复，P/Q 覆盖率为 100%，没有删去失败记录。

公开通用 RiNALMo 权重、冻结状态、表征文件 SHA256 及生成精度见 embedding_completion_TR0_VL0_TS0_bprna_new.json。训练和验证的输入文件哈希见 split_audit.json。三组 smoke test 的初始参数哈希相同，均验证了非零且有限的 head/CNN 梯度，详情见 preparation_complete.json；smoke 分数仅用于运行检查，不是正式实验结果。

评估器补充检查了完整真值中的不可预测配对仍计入召回率分母，并拒绝篡改真值、阈值和非法预测。结果见 evaluation_contract.json。

科学代码及环境在正式训练前固定于提交 `498a183a79d26d3fa6f00aa8247f2ba127309022`，逐文件哈希记录于 experiment_identity.json。正式训练的作业 ID 为 9133–9141，按照 Slurm 可用 GPU 资源调度。最终运行状态及结果以本目录后续的 README_results.md、cluster_status.json 和 artifact_validation.json 为准。
