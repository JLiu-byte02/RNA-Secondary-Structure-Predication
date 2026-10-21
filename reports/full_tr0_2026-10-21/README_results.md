# 完整 TR0 多种子训练与最终测试

九组正式训练及九组最终评估全部完成；19 个 Slurm 作业成功退出，全部 27 份预测通过独立指标复算。检查了 71,658 条逐 RNA 预测，选中 checkpoint 参数与对应 epoch 一致，科学源文件与运行时指纹一致。

| 方法 | 集合 | RNA-macro exact F1，均值 ± 样本 SD | Precision 均值 | Recall 均值 |
|---|---|---:|---:|---:|
| E0_no_thermo | VL0 | 0.611268 ± 0.002361 | 0.615401 | 0.631315 |
| E1_bpp | VL0 | 0.647173 ± 0.002680 | 0.649816 | 0.667944 |
| E2_bpp_stack | VL0 | 0.645788 ± 0.000832 | 0.646510 | 0.668199 |
| E0_no_thermo | TS0 | 0.606643 ± 0.001392 | 0.609423 | 0.625323 |
| E1_bpp | TS0 | 0.646526 ± 0.002507 | 0.650616 | 0.665064 |
| E2_bpp_stack | TS0 | 0.647474 ± 0.001016 | 0.648987 | 0.667366 |
| E0_no_thermo | bprna_new | 0.504694 ± 0.008873 | 0.508212 | 0.526323 |
| E1_bpp | bprna_new | 0.618096 ± 0.009508 | 0.616861 | 0.645934 |
| E2_bpp_stack | bprna_new | 0.626139 ± 0.005850 | 0.616507 | 0.662686 |

使用种子 17、23、41，完整 TR0 10,682 条训练，VL0 1,286 条验证，TS0 1,288 条和 bpRNA-new 5,388 条最终测试。四集合无 ID 或序列交叉。三种方法结构和参数量相同，同一随机种子初始化及训练顺序一致；通用 RiNALMo-Giga 始终冻结，没有载入监督 checkpoint。

每组最多 10 epoch、patience 3。每个 epoch 用 VL0 完整真值 RNA-macro exact F1 选择 checkpoint；并列取较早 epoch。全部训练结束后，每种方法在三个已选 checkpoint 的 VL0 曲线上选择一个共同阈值，并列取较小阈值。`selection_lock.json` 在提交任何测试作业前固定 checkpoint SHA256、epoch 和阈值。TS0 和 bpRNA-new 分数不用于任何选择。

主指标始终对完整 original 配对计算 exact P/R/F1。候选 BCE 训练标签仍沿用原 B0 eligible 策略；不可预测的完整真值配对仍计入召回率分母。所有预测要求合法 canonical 候选、最小间隔 4、每个核苷酸最多一个配对；允许假结。均值及 SD 基于三个独立种子，SD 为样本标准差，不能当作置信区间。

训练源代码提交：`498a183a79d26d3fa6f00aa8247f2ba127309022`。源文件、环境与公开权重来源记录见 experiment_identity.json，训练日志摘要见 training_runs.csv，逐种子结果见 full_results.csv，配对差值见 paired_deltas.csv。各个最佳 epoch、单种子 checkpoint 选择阈值及最终方法共享阈值分别保留，避免混淆。

完整产物保存在服务器 `/home/junxuanliu/projects/RNA_ThermoFusion/outputs/full_tr0/`，包括所有 checkpoint、逐 epoch 日志、阈值曲线和逐 RNA 预测。embeddings/full/ 保存四集合表征；既有 FAST-v1 输出保持原样。GitHub 保存代码、配置及本报告证据，预测的文件位置与 SHA256 见 predictions_manifest.json。

该阶段按用户指定在完整 TR0 上训练，已经不具备 FAST-v1 的 family-dev 监督家族留出条件。VL0 属于选择集，不能作为独立测试结论；两个测试集结果按本阶段锁定协议报告。历史项目记录曾包含测试集评估，本次没有据此改动结构、预算或阈值网格，因此这里是固定协议复跑结果，不能声称这些测试数据在整个项目历史中从未被查看。
