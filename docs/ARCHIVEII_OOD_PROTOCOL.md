# ArchiveII Family-OOD：RiNALMo 对齐协议

本轮独立于此前 TR0 和 FAST-v1 实验。九个测试家族依次为5s、16s、23s、tRNA、srp、tmRNA、RNaseP、grp1、telomerase；E0/E1/E2 每折重新从通用预训练 RiNALMo 初始化，不继承任何此前监督训练参数，沿用种子17、23、41。

## 官方来源与划分

官方实现固定为 [lbcb-sci/RiNALMo 提交2c2c5c1](https://github.com/lbcb-sci/RiNALMo/tree/2c2c5c14a5ae609d8c560a5d9ca32e51e0288955)。论文为 Penić et al., Nature Communications 16, 5671 (2025), [doi:10.1038/s41467-025-60872-5](https://doi.org/10.1038/s41467-025-60872-5)。官方数据下载代码指向 [ct-splits.tar.gz](https://github.com/marcellszi/dl-rna/releases/download/Data/ct-splits.tar.gz)，只采用其中`ct/fam-fold/`九折，保留各折既有train/valid/test，不自行随机重分。源包SHA256记录在配置及审计中。

九个完整测试家族样本数（逐文件核对）：5s=1283、16s=67、23s=15、tRNA=557、srp=918、tmRNA=462、RNaseP=454、grp1=74、telomerase=35，共3865条。16S/23S沿用基准中的独立折叠domain，不用完整长rRNA替换。训练/验证及测试均不新增长度过滤、不裁剪，不用测试家族选择阈值或模型。审计验证每折三个角色的ID互斥、测试家族完全留出、所有折CT内容一致，并单独披露官方数据中潜在序列重复；不擅自去重后仍宣称同一基准。

## 训练规则

沿用官方Lightning包装器及GradualUnfreezing回调：前3轮仅训练预测头；零起始epoch3/6/9/12分别解冻最后3层及最终LayerNorm、再往前3层，保持官方新增参数组的学习率处理。固定15轮、batch1、不梯度累积、不新增梯度裁剪；Adam（非AdamW），初始LR=1e-4；官方LinearLR start=1、end=0.1、total_iters=7000，按optimizer step更新。CLI默认LR=5e-4而论文Methods明确1e-4，因此显式传论文值，并保留这一来源差异。使用16-mixed及官方全严格上三角BCE，真值保留原始配对。解码时才排除非典型配对和尖环。

官方`train_sec_struct_prediction.py`在训练结束后直接`trainer.test(model=model)`，因此本轮主结果使用第15轮参数及最后一次验证调优的阈值，不沿用旧实验的最佳epoch选择。阈值候选为0.01至0.29步长0.01，按验证集逐RNA平均官方F1最大化，相同时取最先的低阈值。每种子独立调阈值，测试前保存锁定记录。

单条轨迹保持完整不中断的15轮流程。若运行中断，保留该次尝试的所有产物并从相同公开预训练权重、相同种子重跑整条轨迹；不把原生回调恢复优化器参数组但未恢复解冻标志的部分续训当成相同过程。监督器可以复用已完整完成且哈希一致的轨迹，之前的中断尝试单独记录。

E系列保持项目PairHead+CorrectionCNN结构及E0/E1/E2物理通道定义，配合逐步解冻后的在线RiNALMo输出。为遵循新协议，本轮预测头不强制BF16，也不在计算全上三角BCE前硬掩码logits。物理特征仍为序列自身ViennaRNA2.6.4 Turner2004、37°C同一次partition function的P及真实stack联合概率Q，不使用结构标签生成特征。官方RiNALMo基线具有原生SecStructPredictionHead；它与E预测头的架构差异是本研究的比较对象，不能把E模型本身称为原始RiNALMo。

## 评估与基线

直接调用固定官方源码`prob_mat_to_sec_struct`及`ss_precision/ss_recall/ss_f1`。主指标保留官方默认`allow_flexible_pairings=True`，允许一个配对端点有1nt偏移；同时报告完全精确的配对F1作为附加指标。每家族先逐RNA计算P/R/F1再取算术平均；九折主汇总为九家族等权均值，另报告按3865条RNA加权的总体均值，二者明确区分。

基线下载[Zenodo官方九个ArchiveII微调权重](https://zenodo.org/records/15043668)，核对公开MD5及本地SHA256，加载各自保存的阈值，在完全相同的测试CT与官方解码/指标上重新评估。报告区分“官方检查点重评估”（每家族一个固定模型）和“E系列三种子均值±样本SD”，不把官方单次模型冒充三种子实验或声称与作者随机轨迹逐位相同。新增多种子重复用于衡量E方法的训练波动，划分、训练策略和评估规则保持固定。

完整实验预计81条E训练轨迹、9组官方检查点评估。保存固定源码指纹、数据成员清单、逐轮验证、最终阈值锁定、逐RNA预测与独立复算证据；全部正式运行前通过数据/指标/解冻/FP16梯度及最长序列显存检查。大文件留在服务器，代码及压缩审计清单、报告持续同步GitHub。

环境采用独立`.venv_archiveii`覆盖层，保留原项目环境。官方pip发行构建未包含完整下游数据模块，因此将该固定提交的33个`rinalmo/`源码文件逐Git blob核对后保存在`third_party/rinalmo_upstream/`，仅本轮入口优先加载；官方算法文件保持原字节，未修改旧环境。完整官方许可证见`third_party/rinalmo_reference/LICENSE`。
