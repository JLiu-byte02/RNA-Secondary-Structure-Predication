# RNA ThermoFusion：快速验证版 README

**版本：** 2026-10-10 · `FAST-v1`  
**任务：** RNA 序列 → RNA 二级结构（碱基配对集合）  
**重点：** 家族留出（family shift）下的精确配对预测  
**原则：** 复用现有 B0 / RiNALMo / 数据 / 解码代码，先验证物理信息有没有增益，再考虑新方法。

> **一句话目标：** 不再训练竞争 stem ranking，而是在当前 RNA 大模型预测的配对分数上加入热力学证据，检验它能不能补回真正的配对、减少错配，尤其是在未见 RNA 家族上。
>
> **第一轮只比较三组：** `E0（不加物理信息） → E1（加入 BPP） → E2（加入 BPP + stack joint probability）`。E2 只有在 E1 值得继续时才运行。**不重新训练 RiNALMo，不做九次参数扰动，不做多轮迭代纠错。**

---

## 1. 现有进度与这轮要解决的问题

2026-10-08 的已完成实验，在相同的冻结 RiNALMo 表征和预测头条件下比较 B0/B1/B2/B3。主表为精确配对 F1：

| 方法 | TS0 | bpRNA-new | 当前处理 |
|---|---:|---:|---|
| RNAfold | 0.5086 | 0.6169 | 物理方法对照 |
| MXfold2 | 0.5693 | 0.6333 | 混合方法对照 |
| B0（原基础模型） | **0.6133 ± 0.0038** | **0.4616 ± 0.0131** | 保留作为已有基线 |
| B1（随机竞争排序） | 0.5820 ± 0.0059 | 0.4400 ± 0.0036 | 暂停 |
| B2（困难配对训练） | 0.5960 ± 0.0046 | 0.4660 ± 0.0086 | 未证明稳健改善 |
| B3（困难竞争排序） | 0.5465 ± 0.0089 | 0.3685 ± 0.0208 | 暂停 |

这些是已有最终测试结果，**不能当作本轮可反复调参的目标**。而且旧 B0 最终 checkpoint 用到了完整 TR0；**不能拿它来评估从 TR0 留出来的 family-dev**。

### 这轮只回答两个问题

1. **BPP 是否有用？** 与同一网络、同一训练协议、不加物理信息的 E0 比较。
2. **stack 联合概率是否有额外用处？** 在 BPP 已经输入的情况下，E2 是否继续优于 E1。

THERMOTOKEN 给了“热力学输入有用”的线索，但论文未单独做 *单张 BPP vs 九张 BPP* 对照，而且它冻结的是**已经接受 RNA 二级结构监督微调的 RiNALMo 及预测头**。所以本项目先从最简单、可复现的 BPP 信息开始，不直接复制其完整方法。

---

## 2. 数据：直接复用现有清洗结果，不重新收集

### 2.1 第一轮只用原有 family-dev 划分

| 集合 | 数量 | 用途 |
|---|---:|---|
| `fit-train` | **3,299 条** | 训练轻量预测头 / adapter |
| `family-dev` | **1,068 条，292 个留出家族** | 选择模型、观察跨家族变化 |
| `VL0` | **1,286 条** | 兼容旧协议；第一轮只做分布内辅助监测，不能替代 family-dev |
| `TS0` | **1,288 条** | 第一轮禁止调参使用 |
| `bpRNA-new` | **5,388 条** | 第一轮禁止调参使用 |

`fit-train` 与 `family-dev` 来自此前 **4,367 条 high-confidence TR0** 的家族划分。原项目另有 **10,682 条合格 TR0**，此时先不要用来训练 fast-v1；等首轮配置确定后再讨论完整重训。

**特别注意：** family-dev 是监督训练的留出集；不能使用包含这些 RNA 的旧“全量 TR0”模型作开发期基础。若当前只有全量 B0 checkpoint，先按已固定的 `fit-train` 重新训练一次 B0；不要用它代替家族开发模型。

### 2.2 单条数据要有的字段

```text
rna_id             唯一 ID
sequence           A/U/C/G（与旧流水线一致的规范化）
base_pairs         真值配对列表，统一使用 0-based (i,j), i<j
family_id          家族标识；缺失时保留 unknown，不凭空补家族
split              fit-train / family-dev / VL0 / TS0 / bpRNA-new
embedding_path     已缓存的 RiNALMo 表征（可选）
```

**不要重新随机划分。** 从原仓库读取既有 split manifest，训练前保存 family-ID、序列去重及 checkpoint 来源检查结果。对无法确认训练来源的旧 checkpoint 不做“严格未见家族”宣称。

### 2.3 第一轮的最小质量检查

- 一条 RNA 的序列长度、配对标签长度、RiNALMo 每核苷酸 embedding 长度完全一致，排除 CLS/EOS 偏移。
- `fit-train` 和 `family-dev` 无重复序列；同一已知家族不跨这两个集合。
- 真值可包含物理模型未覆盖的配对；**不要悄悄删除不被 ViennaRNA 支持的真值来提高指标**。沿用 B0 的标签处理及 decoder 范围，并报告不可解码真值的占比。
- 长 RNA、含非标准碱基或 ViennaRNA 计算失败样本要记录 ID、原因和覆盖率；各实验使用**同一批成功样本**比较，不选择性剔除难例。

---

## 3. 输入：RiNALMo 分数 + 两类热力学特征

### 3.1 原有神经模型

```text
RNA sequence
    │
    ▼
Frozen RiNALMo encoder  ──► cached nucleotide embeddings H
    │
    ▼
Existing B0 pairing head ──► initial pair logits S0 [L, L]
```

- **RiNALMo encoder 一直冻结**，直接读取已有缓存。
- **原来的 B0 配对头原样保留**，不要首轮同时换成新 ResNet/MoE/RAG。
- 使用相同的配对掩码、阈值选择方式和最终解码器，确保新旧差别可解释。

### 3.2 热力学输入 P：一张普通 BPP

用 ViennaRNA 计算参考参数下的 base-pair probability：

\[
P_{ij}=\Pr\big((i,j)\text{ 在热力学折叠集合中出现}\big).
\]

这里不是 RNAfold 的一张 MFE 结构，也不是训练标签，而是**每个候选配对的软概率**。

首版统一配置：ViennaRNA `2.6.x`；温度 `37°C`，`dangles=2`，`noLP=False`，`uniq_ML=True`，默认 Turner 参数。一次 `fold_compound` 的 partition-function 运算生成 P；输出缓存到磁盘。`fc.bpp()` 的返回矩阵在 Python 接口中也需核对 1-based 约定，统一映射到项目的 0-based 坐标。与 THERMOTOKEN 论文的物理默认环境尽量保持可比，但绝不更改真值标签。

### 3.3 热力学输入 Q：相邻配对联合出现的概率

\[
Q_{ij}=\Pr\big((i,j)\text{ 和 }(i+1,j-1)\text{ 在同一结构中出现}\big).
\]

它不是普通配对概率的相乘：`Q[i,j] != P[i,j] * P[i+1,j-1]` 一般不成立。它描述两个相邻配对**一起形成局部堆叠（stack）**的概率，不是整个完整 stem 的概率。

ViennaRNA 的 Python 接口在 `fc.pf()` 之后提供 **`fc.stack_prob(cutoff=...)`**，返回 `(i,j,p)` 记录：`i,j` 表示外层配对，用 **1-based → 0-based** 转换填入 `Q[i-1,j-1]`。这是一份列表，不是对每个 `(i,j)` 独立调用的标量函数。注意检查哨兵项、截止阈值与版本差异。Q 只需在同一参考 ensemble 上计算，**无需再运行九遍 partition function**。

实现时只存上三角，送入网络前按旧代码的矩阵约定镜像；不在无效位置填人工高值。第一轮可对短序列抽样核验：`0 ≤ Q[i,j] ≤ min(P[i,j], P[i+1,j-1]) + 容差`。

> **快速性关键：** 只计算一次参考物理模型，就得到 P 和 Q。先不要实现 THERMOTOKEN 的四组 ± 参数扰动；它的九张图属于后续可选实验。

---

## 4. 模型：只增加一个很小的二级结构纠错器

### 4.1 推荐架构（首轮固定）

```text
                RNA sequence
                     │
             Frozen RiNALMo
                     │
                Existing B0
                     │
              pair logits S0 ──────────────────┐
                                               │
RNA sequence ── ViennaRNA ──► BPP P ──────────┤
                              stack Q ────────┤
                                               ▼
                                 Small 2D correction CNN
                                               │
                                        residual ΔS
                                               │
                                      S = S0 + ΔS
                                               │
                                original B0 decoder
                                               │
                                    predicted base pairs
```

**纠错器 `CorrectionCNN` 的默认设置：**

- 输入通道：`S0`（原配对 logits）、`P`、`Q`、`distance=|i-j|/L`、`pair_valid_mask`，共 **5 通道**；缺失物理特征直接置零。
- `Conv2d(5,32,3,padding=1) → GELU → Conv2d(32,32,3,padding=1) → GELU → Conv2d(32,1,1)`。
- 输出最后一层**零初始化**，初始 `ΔS=0`，起点就是 B0 的预测分数。
- 预测分数 `S= S0 + ΔS`，在有效配对区域内作对称处理；无效位置用与旧 B0 一样的 mask；最终**沿用旧 B0 解码器**。
- 暂不引入跨步 message passing、迭代更新、MoE、排名头，也不依赖新训练数据。

**训练方式（避免训练/测试时 B0 分数分布不一致）：** 第一轮 **只冻结 RiNALMo，B0 head 与 CorrectionCNN 共同训练**。B0 head 如需载入已有权重，只能载入**仅以 fit-train 训练**的开发 checkpoint，并让 E0/E1/E2 使用完全相同的初始化；若没有此权重，就让三组从相同随机初始化开始训练 B0 head，**绝不能载入使用过全量 TR0 的 B0 权重**。所有三组使用同一个架构、训练集与训练轮次。不要先用“在 fit-train 上训练得近乎完美的 B0”产生训练分数，再冻结它单独在这些样本上训练一个融合器，却把这套训练分布直接当成家族开发分布。若仓库已有可靠的 OOF logits，才考虑独立冻结式融合器作为加速备选。

> 为什么不是一开始大改预测头？当前需要先判断新证据是否有帮助；改动越少，效果越容易归因。

### 4.2 三个实验只有物理通道不同

| 实验 | S0 | P | Q | 训练参数/代码 |
|---|---|---|---|---|
| `E0_no_thermo` | 有 | 全零 | 全零 | 与后两组完全相同 |
| `E1_bpp` | 有 | 真实 BPP | 全零 | 与 E0 完全相同 |
| `E2_bpp_stack` | 有 | 真实 BPP | 真实 stack 联合概率 | 与 E0/E1 完全相同 |

通过**相同的 5 输入通道**保证参数数量一致；E0 是“增加同样小的 CNN，但不加物理信息”的公平对照。原始旧 B0 仍单独报告，但**不能仅拿 E1 对旧 B0 证明 BPP 有效**。

---

## 5. 训练：先单种子快测，再确认

### 5.1 训练目标

**原 B0 的配对预测 loss 如果已有可靠实现，首版完全复用，三组不得分别改 loss。** 不重新加入 B1/B2/B3 的竞争 ranking。

如必须新写一个统一 loss，默认使用每条 RNA 的正负类平衡 BCE：

\[
\mathcal{L}=\tfrac12\,\mathrm{mean}_{Y_{ij}=1}\mathrm{BCEWithLogits}(S_{ij},1)
+\tfrac12\,\mathrm{mean}_{Y_{ij}=0}\mathrm{BCEWithLogits}(S_{ij},0).
\]

只对原 B0 支持的有效配对上三角计算；缺失某一类别时省去空项。每条 RNA 先计算 loss，再对样本平均，避免长序列凭借更多负例主导训练。首版**不加新的排名、物理一致性或复杂 soft-F1 损失**，以免混淆信息本身的贡献。

### 5.2 默认训练参数

```yaml
# 建议生成 configs/thermo_fast.yaml
seed: 17
encoder: existing_frozen_rinalmo_cache
head: existing_B0_head
adapter: 2d_cnn_32_channels
adapter_input: [b0_logit, bpp, stack_joint, normalized_distance, valid_mask]
train_split: fit-train             # 3299
select_split: family-dev          # 1068, 292 families
initial_lr: 0.0003
optimizer: AdamW
weight_decay: 0.01
batch_size: 1                    # 一条完整 RNA，长度可变
max_epochs: 10                   # 快测默认；若原 B0 需要更长收敛则统一增加
patience: 3
clip_grad_norm: 1.0
precision: mixed_if_supported
selection_metric: family_macro_exact_f1
threshold: tune_on_family_dev_only
allow_rinalmo_grad: false
use_TS0_for_selection: false
use_bpRNA_new_for_selection: false
```

这是**起始配置**而非已经验证最优的参数。若 B0 现有训练代码有成熟且不同的学习率/预算，优先沿用已有设置，但所有三组必须一致。选择规则固定：family-dev 宏平均 exact F1 最高的非初始 checkpoint，平分时取较早 epoch。不要因为看到 TS0 或 bpRNA-new 结果再改参数。

### 5.3 快速执行次序

1. **Smoke test（只检查程序）：** 用 32–64 条 fit-train 和 16 条 family-dev，1 个 epoch；验证张量维度、反向传播、P/Q 对齐、解码与指标。**这些小样本结果不判断科学有效性。**
2. **正式快测一：** 用完整 `fit-train / family-dev`、seed 17，训练 `E0` 和 `E1`；先只比较这两个。
3. **正式快测二：** 如果 `E1` 对 E0 有可辨识收益，再运行 `E2`；并针对可能的收益补 seed 23、41。无稳定信号则停止，不继续堆模块。

### 建议的 CLI（交给 Codex 创建，而非声称项目已存在这些脚本）

```bash
# 由现有数据清单计算并缓存热力学特征（只需一次）
python scripts/cache_thermo.py --split fit-train,family-dev --features bpp,stack

# 在同一训练设置下，仅控制物理输入通道
python scripts/run_thermo_fast.py --config configs/thermo_fast.yaml --mode E0_no_thermo
python scripts/run_thermo_fast.py --config configs/thermo_fast.yaml --mode E1_bpp
# 看到明确趋势后再跑
python scripts/run_thermo_fast.py --config configs/thermo_fast.yaml --mode E2_bpp_stack

# 同一评估器输出 family-dev 指标与每条 RNA 的预测
python scripts/eval_thermo_fast.py --split family-dev --compare E0_no_thermo,E1_bpp,E2_bpp_stack
```

只需新建或修改 3 个脚本：`cache_thermo.py`、`run_thermo_fast.py`、`eval_thermo_fast.py`；训练、数据读取、B0 head 和 decoder **全部优先复用原仓库代码**，不要重新搭一套流水线。

---

## 6. 第一轮必须看哪些结果

### 6.1 核心结果表（由脚本自动生成）

| 模型 | family-dev 家族宏平均 exact F1 ↑ | RNA 宏平均 exact F1 ↑ | VL0 exact F1 ↑ | 比 E0 提高多少 |
|---|---:|---:|---:|---:|
| B0（历史开发基线，若协议一致） | 待填 | 待填 | 待填 | — |
| E0：无物理输入 | 待填 | 待填 | 待填 | 0 |
| E1：BPP | 待填 | 待填 | 待填 | 待填 |
| E2：BPP+stack | 待填 | 待填 | 待填 | 待填 |

**主要判据：** `family-dev` 的 **逐家族等权 macro exact F1**；同时报告逐 RNA macro exact F1。VL0 只是辅助，不允许把有利的 VL0 分数取代 family-dev 主判断。

### 6.2 两个必要的结构分析（不要做十几种）

- **Pair correction：** 与 E0 相比，E1/E2 补回多少真实配对、删掉多少错误配对，又新增了多少错误配对、删掉多少真配对。
- **Stem recovery：** 用原仓库现成的完整配对片段统计，检验改进是不是发生在连续 stem 上，而不仅仅改变配对总数。

额外记录每个家族的 `ΔF1`、参与评估的 RNA 数目，以及 ViennaRNA/P/Q 成功计算的覆盖率。先不做 SHAPE 数据、假结预测、多目标损失或临床下游。

### 6.3 简单的继续 / 停止规则（开发期经验门槛，非统计显著性声明）

- **积极信号：** E1 比 E0 家族宏平均 exact F1 大约提升 **≥0.015**，逐 RNA 指标同向，且没有靠少数异常家族支撑 → 值得跑 E2 和另两个种子。
- **弱信号：** 仅增加 `0–0.015` → 先检查受益家族、解码后的真配对增减，不直接写成方法成功；最多增加第二种子确认。
- **负信号：** E1 不升反降，且重复后仍下降 → 不急着做九张 BPP；优先确认物理特征对齐、训练设置和预测头起点。
- **Q 的去留：** E2 只有在相对 E1 提供**跨种子一致的额外收益**时才保留；否则采用简单 E1。

注意：门槛仅用于节省计算的项目决策。最后是否有效，要依据多个种子、逐家族差异和预先固定的测试协议。

---

## 7. 若快测有效，下一步只有两件事

**第一件：做 Q 的必要对照。** 比较真实 `Q` 与可由普通 BPP 计算的 `P[i,j] × P[i+1,j-1]`，使用同一个 CNN 和相同预算。否则 Q 的收益可能只是把邻居配对信息显式给了模型，不能说明真正的联合统计有独立价值。

**第二件：冻结方法再做正式测试。** 在所有配置、训练轮次和阈值选择规则决定后，按现有完整训练协议使用 **10,682 条合格 TR0** 训练，执行既有 VL0 阈值选择，然后在 TS0、bpRNA-new 上一次性比较 exact F1、RNAfold/MXfold2 基线及各类 stem/配对修复指标。由于此前已经研究过这两个测试集，论文中不要声称它们从未影响过方法选择；若希望更强泛化论证，后续增加一份新的独立未见家族验证。

**只有完成这两件事且确有稳定收益，才讨论九张扰动 BPP、迭代纠错，或更成熟的局部结构建模。** 它们现在都不是首版必做项。

---

## 8. 实现细节中最容易出错的地方

1. **checkpoint 泄漏：** full-TR0 B0 不能用于 family-dev 的“未见家族”对照；上游任务微调过的权重同样要查训练来源。
2. **索引偏移：** ViennaRNA `stack_prob` 为 1-based；本项目标签与配对矩阵应统一 0-based；Q 的几何位置 `(i,j),(i+1,j-1)` 要核验。
3. **不同 mask：** E0/E1/E2 要用相同的正负候选定义、decoder、阈值搜索规则；不可只让物理模型处理较容易的 RNA。CNN 应读取**施加 -inf mask 之前的有限 S0 logits**，CNN 后才应用无效配对 mask，避免 NaN。
4. **配对标签范围：** 如果真值含模型不允许的假结或非经典配对，应**统一记录**，不要暗中删除标签后仍与旧 F1 比较。
5. **CNN 对称与 padding：** 输入实对称矩阵时需要保证 `S[i,j] == S[j,i]`；padding 不参与 loss/decoder。
6. **Q 不是全局 stem 真值：** 它只描述局部相邻配对的联合概率，不能把它解释为整个 stem 的物理可靠度。
7. **不要凭单次 seed 宣称胜利：** seed 17 只是快筛；进入论文必须补种子、同协议比较和误差分析。

---

## 9. 交付物：Codex 执行后应留下什么

```text
configs/thermo_fast.yaml
scripts/cache_thermo.py
scripts/run_thermo_fast.py
scripts/eval_thermo_fast.py
outputs/thermo_fast/
  split_audit.json                  # 划分与 checkpoint 来源核查
  thermo_cache_manifest.json       # ViennaRNA 版本 / 参数 / 序列覆盖率
  E0_no_thermo/seed17/...          # checkpoint、训练日志、逐条预测
  E1_bpp/seed17/...
  E2_bpp_stack/seed17/...          # 只有明确值得做时生成
  development_summary.csv          # 三组 family-dev 主指标
  family_delta.csv                 # 逐家族表现
  pair_stem_corrections.csv        # 真/错配对与 stem 分析
  README_results.md                # 结论：继续 / 修改 / 停止及理由
```

**完成定义：** 不需要形成论文，也不需要跑全部旧方法；只要完成同一 family-dev 上的 E0/E1 对照，若有趋势再做 E2，并自动导出上面的最小结果，就能回答“物理信息是否值得继续做”。

---

## 10. 论文定位（暂定，不当成已证实结论）

- **应用问题：** RNA 二级结构在未见家族上的迁移预测。
- **已知技术基线：** 冻结 RNA foundation model、BPP 热力学输入和轻量纠错模块。
- **待检验贡献：** 相邻配对联合占据（stack joint probability）能否提供普通 BPP 之外的有效结构证据，改善连续 stem 的恢复。
- **不要宣称：** “首次融合 RNA 大模型与热力学”“首次使用联合配对概率”“已经解决 structure OOD”。这些表述超出目前证据。

### 来源与实现参考

- 当前进度：上传的《实验整体进度简要总结》，2026-10-08。
- THERMOTOKEN: *Thermodynamic Response-Conditioned Adaptation for RNA Secondary Structure Prediction under Family Shift*（上传 PDF；主要参考方法及物理特征）。
- BioSynFold: *Biology-Guided Synergistic Learning of RNA Foundation Models for Secondary Structure Prediction*（上传 PDF；主要参考冻结大模型与轻量结构预测器）。
- ViennaRNA 官方 Python API：<https://viennarna.readthedocs.io/en/latest/api_python.html>（`fold_compound`、`pf`、`bpp`、`stack_prob`）。

**最终执行建议：现在先做 E0 和 E1。只要 E1 有可信的家族迁移增益，才花计算预算做 Q 和后续创新。**
