# ArchiveII冻结E：准备阶段验收

2026-10-22：官方九折CT及P/Q、公开通用RiNALMo冻结表征完成准备。正式结果仍待81条50轮轨迹及全部测试完成，不在此处填入尚未测得的性能。

- 用户确认：保留原E的冻结编码器和50轮设置，使用种子17、23、41，仅替换为官方ArchiveII九折数据。
- 数据包SHA256、3865条RNA、3354条唯一序列及九折成员/CT字节一致性审计通过；详见split_audit.json。
- 5个数据/指标/解码/配置与namespace来源回归测试，以及3个既有模型/AdamW续训测试通过。
- 真实GPU上E0/E1/E2各完成小样本2轮smoke，所有梯度及共同完整初始化验证通过。这些smoke分数不是正式结果。
- 最长512nt训练RNA完成E2前向/反向检查；公开通用权重、冻结表征及9个官方数值模块Git blob核对通过。
- 首次表征来源记录因namespace package没有__file__而失败；修复并重新准备。失败9218、依赖检查9219保留审计，成功准备9220、smoke9221均为COMPLETED、ExitCode0:0。2026-10-23更正：9219此前误记“未启动即取消”，实际Slurm为FAILED/1:0，因缺少embedding_provenance.json失败；更正依据和原记录见historical_preparation_correction.json。
- 训练数值源码封存SHA256：`30335ca3a29df8256d2ae3584d6ebfb8133792c38ced660264358dd1e86f0d41`。
- 输入封存SHA256：`2538b9f11fa22f82f8be2fdaaa86563e8a2c690d5772f74762c58cebc4dac517`。
- 正式训练源代码提交：`40bd9aff9e144758ba3d34b8bef454ae0065ead9`。

实验身份和实际准备证明随此文件提供。正式训练以独立新命名空间执行，不继承之前已取消的在线微调轨迹或TR0监督检查点。官方RiNALMo基线权重下载/校验和后续测试状态以服务器运行记录为准。
