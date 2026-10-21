# 长轮数报告和图表契约

研究问题：增加训练预算是否改善验证和测试性能，改善幅度是否与时间成本相称，B0 与 E 系列有何差异。观察结论仅由完成后的真实数据计算，绘图代码不生成模拟数据。

`scripts/report_long_epochs.py` 必须等全部 12 条 50 轮轨迹、60 个预算选择、12 个测试作业以及独立预测复核完成后才生成最终报告。报告保留 600 条逐轮记录、180 个预算/方法/种子/集合结果、20 个三种子汇总以及逐种子收敛指标。

`scripts/plot_long_epochs.py` 再检查完整数据网格和测试结果聚合后导出以下单图。每张图对应明确的证据作用，逐轮图使用所有种子和轮次，不平滑、不剔除异常值。

| 图 | 证据作用 |
|---|---|
| validation_f1_by_epoch | 轮末验证分数，揭示晚期提升或回落 |
| validation_best_so_far | 预算内可选最优分数的包络；单调性由选择规则产生 |
| training_loss_by_epoch | 在线优化是否继续降低训练损失 |
| validation_loss_by_epoch | 轮末参数在独立验证集的损失变化，与训练损失联合判断过拟合 |
| TS0_f1_vs_budget | 五个预定预算的 TS0 泛化效果 |
| bprna_new_f1_vs_budget | 五个预定预算的 bpRNA-new 泛化效果 |
| validation_vs_time | 预算选择得到的验证收益与累计训练段时间 |
| TS0_paired_vs_B0 | 50轮固定预算，按同种子比较 E 与 B0 的 TS0 差值 |
| bprna_new_paired_vs_B0 | 同规则比较 bpRNA-new 的配对差值 |

中心和误差使用种子17、23、41的均值与样本SD。逐轮图另保留每个种子的细线；配对图显示每个种子的差值和差值的均值、SD。不把SD称为置信区间，不从三种子宣称显著性。最终预算图展示的是各预算内仅按VL0选出的模型，逐轮验证图展示当轮模型，两者含义不同。

时间为实测完整轮训练段，不含排队和验证诊断；E系列包含其已经完成且通过一致性检查的历史前缀成本。检查点续训前后的完整轮耗时均保留，中断部分轮、重新加载及短暂重叠的资源预留另见全部尝试的Slurm分配时长。不同节点负载和GPU状态可能影响计时，不能单凭墙钟差异推断算法效率。候选accuracy和balanced accuracy也在源CSV中保留，主指标仍是完整真值结构F1。

图表用于项目报告，采用Python/Matplotlib，183×118 mm，正文字号9pt、附注7.5pt，输出可编辑SVG/PDF和300dpi PNG预览。没有期刊投稿要求，因此不生成TIFF。源代码预检对“PNG不是TIFF”和“300而非默认600dpi”的提醒符合此报告导出契约。

可选环境变量 `RNA_FIGURE_QA_DIR` 指向已安装的 nature-figure 检查脚本目录。本次交付使用该目录执行最终PDF字体、碰撞检查及绘制后的布局检查，并人工检查所有PNG。全部图只有一个绘图区，可比面板对齐为不适用；若出现碰撞警告，逐项确认并记录。`figure_manifest.json` 保存实际源数据和图表哈希，`figure_contract.json` 在渲染前保存真实数据得出的观察结论。自动检查通过不能代替人工审阅。

复现图表：

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_long_epochs.py --report reports/long_epochs_2026-10-21
```

训练和测试数值代码在正式训练前冻结；后续报告格式及绘图代码不改变数值轨迹。
