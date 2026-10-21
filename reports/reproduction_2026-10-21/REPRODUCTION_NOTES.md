# 2026-10-21 新服务器复现记录

服务器工作目录：`/home/junxuanliu/projects/RNA_ThermoFusion`。运行状态已达到 `completed / validated`，9 组正式实验全部完成，9 组全部通过产物验收。当前结果来自新服务器实际运行，历史汇报原文单独保留。

## 执行与修复

1. 复用并核验项目内 Python 3.11.9、PyTorch 2.1.2+cu118、FlashAttention 2.3.2、ViennaRNA 2.6.4、NumPy 1.26.4 环境。`pip check` 无冲突，CPU、GPU 与真实 Triton rotary-forward 检查均通过。
2. 从本机历史交付包复制 FAST-v1 源码和固定划分。旧代码硬编码 Astra 绝对路径，在新根目录的 preflight 实际触发断言；改为当前源码根目录，并保留原 B0 模块字节哈希检查。
3. 按本次要求重新下载公开通用 RiNALMo-Giga 权重并生成表征。Zenodo 直连较慢，改用官方源码列出的 Google Drive 入口；整文件 SHA256 为 `cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db`，与历史来源一致。未从旧服务器迁移缓存。
4. 首次 GPU 生成作业 9120 因 PATH 缺少 `ldconfig` 退出。系统中实际存在该工具，修正项目激活脚本的 PATH，并将 Triton 与临时缓存放入项目内。重试作业 9121 完成全部表征、划分审计、P/Q 校验和三组 smoke test；独立检查作业 9122 验证实际 rotary kernel 编译与运行。
5. 运行 E0/E1 seed 17 正式对照，通过预定门槛后执行 E2 和共同种子 23、41。9 组正式 Slurm 作业 9123–9131 全部以 `COMPLETED / 0:0` 结束。早停规则保持不变，实际训练轮数为 E0 的 6/8/10 轮、E1 与 E2 各种子的 8 轮。
6. 全量重算逐 RNA 指标、家族聚合、配对修复与 stem 恢复，逐个验证 best checkpoint 等于被选 epoch，并检查全部预测 ID、真值、合法候选、单配对伙伴和指标一致性。9 组验收全部通过，共核对 21,186 条正式预测记录。

## 数据与公平性

固定 fit-train 3,299 条、family-dev 1,068 条 / 292 家族、VL0 1,286 条。训练与开发之间的样本 ID、序列哈希及已知家族交叉均为 0；表征和 P/Q 覆盖率均为 100%。全部 5,653 条冻结表征新生成，未载入任何旧监督 checkpoint。各组同一种子初始权重哈希一致，可训练参数量均为 186,306。

模型、数据与训练规则沿用 FAST-v1；正式训练开始后的科学源码未修改，全部 code_identity 记录与交付源码已再次核对。为适配 CentOS 7 / glibc 2.17，本次使用的 PyTorch/CUDA 环境与历史结果包不同；本报告保留新运行的实际指标和实际早停轮数。

## 结果与边界

E0/E1/E2 的家族宏 F1 均值分别为 0.517469、0.624240、0.623861。BPP 在三个种子上均改善，平均增益 0.106771；Q 相对 E1 的增益为 +0.001420、+0.000545、−0.003102，选择 `prefer_E1`。按预定条件跳过边缘概率乘积对照。

E1 相对 E0 的三种子配对计数汇总：补回真配对 20,567，误删真配对 6,483，删除错配对 25,815，新增错配对 27,799。完整 stem 恢复由 7,124/25,080 提高至 9,838/25,080。F1 的改善不能解释为错误配对总量下降。

所有评价保留完整原始真值，包括无法解码的非经典配对与短间隔配对。family-dev 同时用于 checkpoint 和阈值选择，因此仍是开发结果；三个种子的 SD 不是置信区间。不将监督家族留出解释为 RiNALMo 预训练从未接触过相关序列或家族。本次完成所给报告对应的 FAST-v1 阶段，未扩展全量 TR0 重训、最终测试或九张扰动 BPP。

## 证据与恢复

本目录保存指标、决策、逐家族变化、配对/stem CSV、环境与权重来源、代码哈希，以及完整审计清单的 gzip 副本。`artifact_validation.json` 中 `passed` 为 true，`cluster_status.json` 为 `completed / validated`。完整 checkpoint、逐 RNA 预测、embedding、P/Q 缓存和运行日志保存在服务器 `outputs/thermo_fast/`、`embeddings/`、`logs/`。

```bash
cd /home/junxuanliu/projects/RNA_ThermoFusion
source activate.sh
python -u scripts/reproduce_cluster.py
```

已经完成且配置、输入和源码一致的单组运行会核验后直接复用；中断训练从最近完整 epoch 恢复。失败的 Slurm 作业记录保留在 `previous_jobs`，修复原因后再次启动入口会重新提交。
