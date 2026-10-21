# FULL-TR0-v1

本阶段按用户要求完成 E0（无物理特征）、E1（BPP）、E2（BPP + 真实 stack 联合概率）在完整 TR0 上的训练，以及 VL0 验证和两个固定测试集的评估。三组均独立从随机初始化训练，配对种子为 17、23、41；不载入 FAST-v1 或其他监督 checkpoint。

| 集合 | 条数 | 用途 |
|---|---:|---|
| TR0 | 10682 | 所有训练 epoch |
| VL0 | 1286 | checkpoint、early stopping、最终阈值选择 |
| TS0 | 1288 | 选择锁定后的最终测试 |
| bprna_new | 5388 | 选择锁定后的最终测试 |

所有集合沿用原始 `split_ids.json` 的顺序与成员，四集合间无相同 ID 或相同序列。本阶段不再使用 family-dev 留出；完整 TR0 包含该集合及原先 FAST-v1 未纳入训练的记录。冻结的通用 RiNALMo-Giga、B0 PairHead、纠错 CNN、候选 BCE、贪心解码及真值口径保持不变；监督训练仍在原候选 eligible 配对范围上计算 BCE，所有主评估使用完整 original 配对。

## 训练和选择

配置为 `configs/full_tr0.yaml`。保留 FAST-v1 的 AdamW、学习率 1e-4、weight decay 1e-4、batch size 1、累积 4、梯度裁剪 1、最多 10 epoch、patience 3 及 CUDA BF16。每个种子的三组模型具有相同参数量、初始参数哈希、训练顺序及训练预算。沿用预定的 0.1 至 0.9 阈值网格、严格 `>` 解码比较、阈值并列取较小值、epoch 并列取较早者。

每个 epoch 根据 VL0 RNA-macro exact F1 的最佳阈值评分选择 checkpoint 和 early stopping。全部九组训练结束后，在各组已选 checkpoint 上重新计算 VL0 完整阈值曲线。每种方法使用三个共同种子平均 VL0 F1 最大的**同一个最终阈值**。`selection_lock.json` 固定每组 checkpoint SHA256、最佳 epoch 和最终阈值；之后才提交测试作业。训练不读取测试 embedding、P/Q 矩阵或测试标签；固定 manifest 和全部缓存清单的哈希作为输入版本标识。

这一安排沿用原项目多种子统一阈值的约定。完整 TR0 阶段以 VL0 选择，FAST-v1 以家族留出选择，因此两阶段的开发分数不直接作为同一实验比较。

## 数据准备和执行

仅使用已验证公开权重，SHA256 为 `cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db`。TR0 内已有的新服务器公开权重表征在校验来源及文件哈希后复用，其余表征新生成；全阶段表征单独保存到 `embeddings/full/`。P/Q 继续调用原 ViennaRNA 2.6.4 的同一 PF 实现，已有按序列哈希存储的缓存经参数、矩阵及校验和检查后复用，新缓存保存在 `outputs/full_tr0/thermo_cache/`。要求固定集合 100% 成功，不能静默丢弃失败样本。

```bash
cd /home/junxuanliu/projects/RNA_ThermoFusion
source ./activate.sh
python tests/test_full_protocol.py
nohup python -u scripts/reproduce_full_tr0.py > logs/full_cluster.log 2>&1 < /dev/null &
```

Slurm 准备作业完成四集合 embedding、划分和原标签策略审计、P/Q 缓存及三组 smoke test。随后九组正式训练并行排队，全部完成后锁定选择，再运行九组最终评估。单组训练从最近完整 epoch 恢复；运行期间科学代码的完整指纹必须一致。作业失败后保留错误记录，修复原因后重启 supervisor，失败作业转入 previous_jobs；不得以换参数覆盖既有正式实验。

输出在 `outputs/full_tr0/`：每组 checkpoint、逐 epoch 日志、VL0 阈值曲线、锁定选择、VL0/TS0/bpRNA-new 逐 RNA 预测、每组种子结果、三种子均值和样本标准差。最终校验重新计算全部逐 RNA 指标，核对完整真值、候选合法性、唯一配对、集合顺序、checkpoint 和阈值锁定；只有 `artifact_validation.json` 的 `passed: true` 与 supervisor 的 `completed` 同时满足，才视为完成。

大文件保留在服务器。代码、配置、说明和汇总证据分阶段同步到 GitHub，运行的源代码指纹保留在各组 `code_identity.json` 中。
