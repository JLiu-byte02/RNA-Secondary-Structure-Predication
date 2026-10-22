# RNA ThermoFusion

## ArchiveII Family-OOD（进行中）

独立执行官方ArchiveII九家族leave-one-family-out实验，直接沿用RiNALMo发布的`ct/fam-fold`训练/验证/测试文件。E0/E1/E2各使用种子17、23、41，共81条完整15轮轨迹；每折均从通用预训练RiNALMo重新开始，采用官方逐步解冻、Adam与学习率日程、全上三角BCE及验证阈值搜索。主指标采用官方允许1nt配对偏移的RNA-macro F1，另报告exact F1；九个官方微调检查点在同一测试文件上重评估。具体规则、官方CLI与论文参数差异及单轨迹中断处理见 [ArchiveII协议](docs/ARCHIVEII_OOD_PROTOCOL.md)。

```bash
source ./activate.sh
# Existing Torch/FlashAttention environment is reused without modification.
python -m venv --system-site-packages .venv_archiveii
.venv_archiveii/bin/python -m pip install -c setup/archiveii-constraints.txt -r setup/requirements-archiveii.txt
# Download the exact official ct-splits.tar.gz to data/archiveii_source/ first.
.venv_archiveii/bin/python scripts/archiveii_data.py --prepare
mkdir -p logs
nohup .venv_archiveii/bin/python -u scripts/reproduce_archiveii.py > logs/archiveii-cluster.log 2>&1 < /dev/null &
```

前提是通用RiNALMo权重已按下文完成下载与核验。入口先执行官方源码/环境、指标和GPU梯度预检，再按5s、16s、23s、tRNA、srp、tmRNA、RNaseP、grp1、telomerase顺序入队。Slurm滚动调度，最多9个E训练/基线任务处于提交或运行状态；独立折之间可以重叠，每个模型仍独立初始化并仅使用自己一折的数据。`outputs/archiveii_ood/cluster_status.json`记录阶段与作业；只有状态completed且独立artifact_validation通过才表示完整结束。最终报告计划生成于`reports/archiveii_ood_2026-10-22/`，完成前不将中间结果当作九折最终平均。

## B0 / E 系列长轮数研究（已完成）

固定50轮，比较10、20、30、40、50轮预算，纳入原始纯PairHead基线B0与E0/E1/E2，均使用种子17、23、41和相同TR0/VL0/TS0/bpRNA-new。逐轮记录结构F1、在线训练BCE、VL0 BCE、候选accuracy/balanced accuracy及耗时；统一锁定全部VL0选择后再测试。方案和公平性约束见 [长轮数协议](docs/LONG_EPOCHS_PROTOCOL.md)。本流程接续已完成的全量TR0阶段，E保留原轨迹和优化器状态，B0从头训练。

2026-10-22 已完成全部12条50轮轨迹及固定预算测试，25个最终Slurm作业正常退出。600条逐轮记录、180组预算评估通过独立复核；完整表格、损失、收敛、资源成本及9张图见 [50轮研究报告](reports/long_epochs_2026-10-21/README_results.md)。以下为50轮预算内仅按VL0选择的模型，完整真值RNA-macro exact F1，三种子均值 ± 样本SD。

| 方法 | TS0 | bpRNA-new |
|---|---:|---:|
| B0 | 0.592580 ± 0.005368 | 0.447211 ± 0.017243 |
| E0 | 0.606881 ± 0.003306 | 0.490575 ± 0.015918 |
| E1 | 0.647345 ± 0.001965 | 0.608500 ± 0.012973 |
| E2 | 0.646372 ± 0.001406 | 0.607976 ± 0.015212 |

30→50轮没有改变任何种子的所选检查点，平均每种子额外训练45–59分钟。所有12条轨迹都出现训练BCE下降、验证BCE上升及轮末F1从峰值回落；增加轮数没有带来持续的泛化改善。10→50轮的bpRNA-new所选模型F1也均下降，说明VL0上的小幅选择收益未转化为该测试集收益。此结论限于当前固定学习率与训练协议，三种子SD不代表置信区间。

```bash
source ./activate.sh
mkdir -p logs
nohup python -u scripts/reproduce_long_epochs.py > logs/long_cluster.log 2>&1 < /dev/null &
cat outputs/long_epochs/cluster_status.json
```

最终表格及报告由入口生成于 `reports/long_epochs_2026-10-21/`，原始模型和逐RNA预测保存在 `outputs/long_epochs/`。完成状态需同时满足 supervisor 的 `completed` 与 `artifact_validation.json` 的 `passed: true`。

新环境应先完成下文的环境安装、FAST-v1准备及全量TR0流程，使完整输入和E系列历史检查点可用，再运行长轮数入口。报告图表使用独立的 `requirements-plot.txt`，运行 `python scripts/plot_long_epochs.py`；图表定义和复现说明见 [图表契约](docs/LONG_EPOCHS_FIGURES.md)。

## 全量 TR0 多种子实验

2026-10-21 已完成独立的 `FULL-TR0-v1` 流程：TR0 10,682 条训练，VL0 1,286 条验证，TS0 1,288 条及 bpRNA-new 5,388 条最终测试。E0/E1/E2 均使用种子 17、23、41，19 个 Slurm 作业全部成功，71,658 条逐 RNA 预测通过独立复算。完整结果、checkpoint 和源代码验收证据见 [全量结果报告](reports/full_tr0_2026-10-21/README_results.md)。协议、数据角色及恢复方法见 [全量实验说明](docs/FULL_TR0_PROTOCOL.md)。既有 FAST-v1 的源代码、输出和报告保留。

| 方法 | TS0 exact F1，均值 ± 样本 SD | bpRNA-new exact F1，均值 ± 样本 SD |
|---|---:|---:|
| E0，无热力学信息 | 0.606643 ± 0.001392 | 0.504694 ± 0.008873 |
| E1，BPP | 0.646526 ± 0.002507 | 0.618096 ± 0.009508 |
| E2，BPP + stack | 0.647474 ± 0.001016 | 0.626139 ± 0.005850 |

主指标对完整 original 配对计算 RNA-macro exact F1。最终阈值仅由三种子平均 VL0 F1 选择：E0 为 0.1，E1/E2 为 0.2，在提交测试作业前锁定。BPP 相对 E0 在两套测试集平均提高约 3.99 和 11.34 个百分点；E2 相对 E1 的均值增加约 0.095 和 0.804 个百分点，TS0 的逐种子增益为两正一负，bpRNA-new 为三个种子均正。三个种子的样本 SD 不能视作置信区间。

```bash
source ./activate.sh
mkdir -p logs
nohup python -u scripts/reproduce_full_tr0.py > logs/full_cluster.log 2>&1 < /dev/null &
cat outputs/full_tr0/cluster_status.json
```

## 已完成的 FAST-v1 开发实验

冻结 RiNALMo-Giga 表征，在原 B0 配对预测头上联合训练一个热力学残差纠错 CNN，使用固定家族留出划分比较 E0（无物理信息）、E1（BPP）和 E2（BPP + 真实 stack 联合概率）。历史报告见 [RNA_ThermoFusion_Result.md](RNA_ThermoFusion_Result.md)。2026-10-21 已在新服务器独立完成三种子共 9 组正式实验，并通过全部产物验收；详见 [本次复现报告](reports/reproduction_2026-10-21/README_results.md) 和 [执行说明](reports/reproduction_2026-10-21/REPRODUCTION_NOTES.md)。

| 方法 | family-dev 家族宏 exact F1，均值 ± 样本 SD |
|---|---:|
| E0，无热力学信息 | 0.517469 ± 0.001853 |
| E1，BPP | **0.624240 ± 0.000322** |
| E2，BPP + stack | 0.623861 ± 0.002456 |

BPP 平均增加 0.106771（10.68 个百分点），三个共同种子均改善。Q 的额外收益为两正一负，按预定规则保留 E1。这里报告新运行的实测数值；family-dev 同时参与模型选择，属于开发结果。

模型为 `RiNALMo H[L,1280] → B0 PairHead → S0`，纠错器读取 `[S0,P,Q,|i-j|/L,valid_mask]`，使用 `Conv(5,32,3) → GELU → Conv(32,32,3) → GELU → Conv(32,1,1)` 生成残差，再对称化、掩码和贪心解码。三组均为 186,306 个可训练参数，只有物理通道不同。RiNALMo 不参与监督训练。

固定数据：fit-train 3,299 条，family-dev 1,068 条 / 292 家族，VL0 1,286 条。真值使用完整原始配对。family-dev 用于模型和阈值选择；VL0 仅辅助监测；FAST-v1 不执行 TS0、bpRNA-new 最终测试、全量 TR0 重训或九张扰动 BPP。

## 环境

已适配 CentOS 7 / glibc 2.17 集群：Python 3.11.9、PyTorch 2.1.2+cu118、NumPy 1.26.4、FlashAttention 2.3.2、ViennaRNA 2.6.4。具体版本、安装记录及校验工具见 `setup/`。RiNALMo 官方源码固定在 `setup/rinalmo_commit.txt` 中的 revision。

```bash
cd /home/junxuanliu/projects/RNA_ThermoFusion
source ./activate.sh
python -m pip check
python setup/check_environment.py
```

新机器需要先创建项目内 `.venv`、按 `setup/requirements.txt` 安装科学依赖，再安装对应 CUDA 的 Torch 和 FlashAttention，以及官方 RiNALMo 固定版本。`setup/install_runtime.sh` 和 `setup/finish_install.sh` 保留了本集群的实际安装命令；其中路径和 Anaconda 入口须按目标机器调整。

## 完整复现

```bash
# 登录节点运行；凭据不进入脚本、配置或日志。
source ./activate.sh
mkdir -p logs
nohup python -u scripts/reproduce_cluster.py > logs/cluster_reproduction.log 2>&1 < /dev/null &
```

入口自动执行：官方权重下载及 SHA256 核验 → GPU 生成冻结表征 → 划分及来源审计 → 契约测试 → 一次 partition function 生成 P/Q → 三组 smoke test → 完整 seed 17 E0/E1 → 依据预定门槛扩展 E2 和共同种子 23、41 → 指标、配对修复和 stem 分析 → checkpoint 与全部预测复核。训练作业通过 Slurm 分配 GPU，保持同一随机种子的初始化和训练协议相同。

权重来自 [RiNALMo 官方下载记录](https://zenodo.org/records/15043668)或官方源码中的 Google Drive 入口，仅使用通用预训练权重；SHA256 必须为 `cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db`。设置 `RNA_WEIGHT_SOURCE=google-drive` 可选择后者，下载记录保存实际来源。下载分片和 embedding 分块支持恢复；训练从最近完整 epoch 的 `last.pt` 恢复。失败后查看日志并修复原因，再重新运行入口；失败的 Slurm 作业应在 `outputs/thermo_fast/cluster_status.json` 中核对并单独重新提交，保留既有结果。

```bash
cat outputs/thermo_fast/cluster_status.json
squeue -u "$USER"
tail -n 30 logs/cluster_reproduction.log
# 单组恢复 / 独立运行
sbatch slurm/train.sbatch E1_bpp 17
# 已有输入后使用原始串行入口
python -u scripts/run_thermo_fast.py --pipeline
```

`slurm/prepare.sbatch`、`slurm/train.sbatch` 请求 `g-3090-2` 分区；其他集群需修改分区与资源参数。单独检查核心契约：

```bash
mkdir -p tmp
python tests/test_contract.py
python tests/test_portability.py
```

## 输出与证据

`outputs/thermo_fast/` 保存 split_audit、热力学缓存清单、每组每种子的 checkpoint、逐 epoch 日志、逐 RNA 预测、development_summary.csv、family_delta.csv、pair_stem_corrections.csv、README_results.md 和 artifact_validation.json。完成状态须以 `cluster_status.json` 的 `completed` 和 `artifact_validation.json` 的 `passed: true` 为准。

大体积依赖、权重、表征、热力学缓存、checkpoint 和运行日志留在服务器，不进入 Git。固定 manifest 的压缩副本、split 和来源哈希随代码保留；入口调用 `scripts/unpack_data.py` 校验并恢复 manifest。原始来源记录中的旧路径只用于追溯，不参与新服务器的运行。

family-dev 是开发集，重复种子的标准差不能视作置信区间；监督训练的家族隔离不等于 RiNALMo 预训练从未见过相关序列或家族。P 来自 ViennaRNA BPP，Q 来自同一 ensemble 的真实 `vrna_stack_prob`，不是边缘概率乘积。RiNALMo 官方源码与权重的授权说明见 [官方仓库](https://github.com/lbcb-sci/RiNALMo)。
