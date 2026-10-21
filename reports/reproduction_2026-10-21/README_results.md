# RNA ThermoFusion FAST-v1 实际结果

工作根目录：`/home/junxuanliu/projects/RNA_ThermoFusion`。以下仅纳入本目录实际完成并通过核验的运行。

主指标使用**完整原始真值**的逐家族等权 exact F1，包含 B0 无法解码的配对。未使用 TS0 或 bpRNA-new 预测指标。

|方法|seed|家族宏 F1|RNA宏 F1|VL0 RNA宏 F1（辅助）|最佳/实际epoch|阈值|
|---|---:|---:|---:|---:|---:|---:|
|E0_no_thermo|17|0.515800|0.505609|0.553288|3/6|0.1|
|E0_no_thermo|23|0.517144|0.512347|0.562808|5/8|0.2|
|E0_no_thermo|41|0.519462|0.501834|0.557580|7/10|0.1|
|E1_bpp|17|0.624549|0.603822|0.612639|5/8|0.2|
|E1_bpp|23|0.623906|0.599655|0.614351|5/8|0.2|
|E1_bpp|41|0.624265|0.597006|0.607622|5/8|0.1|
|E2_bpp_stack|17|0.625968|0.604572|0.612710|5/8|0.2|
|E2_bpp_stack|23|0.624451|0.602131|0.614498|5/8|0.2|
|E2_bpp_stack|41|0.621163|0.595950|0.604759|5/8|0.1|

|方法|跨种子家族宏均值±样本SD|RNA宏均值±样本SD|
|---|---:|---:|
|E0_no_thermo|0.517469 ± 0.001853|0.506597 ± 0.005325|
|E1_bpp|0.624240 ± 0.000322|0.600161 ± 0.003436|
|E2_bpp_stack|0.623861 ± 0.002456|0.600884 ± 0.004444|

继续/停止判定：`positive`；Q保留判定：`prefer_E1`。逐种子证据见 decision.json。0.015 是预定计算预算门槛，并非统计显著性。

|方|补回真配对|删除错配对|新增错配对|误删真配对|完整stem恢复/参考|
|---|---:|---:|---:|---:|---:|
|E0_no_thermo|0|0|0|0|7124/25080|
|E1_bpp|20567|25815|27799|6483|9838/25080|
|E2_bpp_stack|21479|25307|30662|5993|10105/25080|

配对变化按同种子E0逐RNA比较，表中为各种子计数之和。Stem复用 maximal_positive_stems(min_stem=3)，参考为完整真值中的连续最大片段；完整恢复要求每个配对均正确，部分恢复为stem内配对召回。

固定数量：fit-train3299、family-dev1068/292家族、VL01286、TS01288、bpRNA-new5388。审计见split_audit.json与provenance/preflight.json，包含去重、家族隔离、CLS/EOS、实际预训练权重SHA256和embedding来源。VL0全部1286条家族unknown，保留并报告RNA宏，不声称VL0家族宏。

特征覆盖率：{"fit-train": {"requested": 3299, "successful": 3299, "failed": 0}, "family-dev": {"requested": 1068, "successful": 1068, "failed": 0}, "VL0": {"requested": 1286, "successful": 1286, "failed": 0}}。逐家族覆盖率见family_delta.csv；失败原因见thermo_cache_manifest.json。

热力学：ViennaRNA2.6.4，37°C/dangles2/noLP=False/uniq_ML=True/Turner2004，cutoff1e-8；一次PF取得P/Q。Python无stack_prob包装，使用同版本共享库vrna_stack_prob；1-based和i=0哨兵经独立短序列全结构枚举核验。Q是真实联合概率，product只作为明确标记的条件对照。

统一训练：原B0 head与候选BCE，CNN末层全零，head与CNN联合训练，encoder冻结。
沿用旧AdamW lr1e-4/wd1e-4/accumulation4/batch1/clip1/BF16/max10/patience3。
Thresholds0.1..0.9: select only by family-dev full-truth family macro F1; lower threshold and earlier epoch break ties.
每条RNA完整输入，无padding。同种子各模式同初始化hash、顺序、预算。
运行成本见development_summary.csv和thermo_cache_manifest.json，wall time含冷启动。
All experiment statuses are explicitly recorded in experiment_status.csv.
Smoke只验证程序。旧历史B0不作本轮公平比较基线。
限制：同一family-dev用于选模型，开发结果不能替代独立最终测试。种子SD不是置信区间。
监督家族留出不代表RiNALMo预训练没有见过相关RNA。
本次不扩展全量TR0重训、最终测试或九张扰动BPP。

恢复：在Astra运行 `. ./env.sh`，再运行 `.venv/bin/python -u scripts/run_thermo_fast.py --pipeline`。
源码/输入哈希见implementation_manifest.json和各组code_identity.json、input_provenance.json。
日志、PID、退出状态见logs/和pipeline_status.json。进度见PROGRESS.md。
