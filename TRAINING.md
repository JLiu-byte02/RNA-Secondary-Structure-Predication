# 训练准备与命令

训练流程与超参数见 [RESULTS.md](RESULTS.md#训练方法)。以下命令从仓库根目录执行。

## 环境与数据

- 安装 PyTorch、NumPy、PyYAML 和 ViennaRNA 的 Python 绑定（导入名 `RNA`），并准备可运行 RiNALMo-Giga 的设备。完整配置在 CUDA 上启用 BF16。
- 安装 [RiNALMo](https://github.com/lbcb-sci/RiNALMo)，并准备 Giga 预训练权重。默认路径为 `weights/rinalmo_giga_pretrained.pt`，可在 YAML 的 `model.backbone.pretrained_path` 中修改。
- 准备 ArchiveII 各折的 `data/datasets/archiveii/<family>/{train,valid,test}.jsonl.gz`，以及 bpRNA 的 `data/datasets/official_bprna_{train,valid,test,bprna_new}.jsonl.gz`。这些路径也可在 YAML 的 `data` 项修改。参考配对是从 0 开始的坐标，并满足 `i < j`。

仓库提供数据转换和缓存生成脚本。训练数据、预训练权重、热力学缓存和训练检查点需要单独准备。

### 准备数据与 BPP

如果已经有符合 YAML 的 JSONL 数据，可以直接生成 BPP。若使用原始输入，[数据构建脚本](scripts/build_datasets.py)需要原始 bpRNA manifest、split IDs 和 ArchiveII CT 划分文件。下例中的三个输入路径按本地文件位置调整。

```bash
python scripts/build_datasets.py --manifest inputs/manifest.tsv.gz --split-ids inputs/split_ids.json --archive inputs/ct-splits.tar.gz --out data
python scripts/cache_thermo.py --inputs data/datasets --out data/thermo/all --max-len 4096
```

### 训练与测试

```bash
python scripts/train.py --config configs/folds/method_bpp_5s.yaml
python scripts/evaluate.py --checkpoint runs/method_bpp_5s/best.pt --config configs/folds/method_bpp_5s.yaml --split 5s

python scripts/train.py --config configs/ablations/tr0_thermo_s17.yaml
python scripts/evaluate.py --checkpoint runs/tr0_thermo_s17/best.pt --config configs/ablations/tr0_thermo_s17.yaml --split TS0 --split bpRNA-new
```

输出写到 `runs/<run_name>/`，包括检查点、训练日志、验证集选择记录和评估指标。ArchiveII 的权重平均入口为 [`scripts/soup.py`](scripts/soup.py)，支持单序列和 FASTA 输入的推理入口为 [`scripts/predict.py`](scripts/predict.py)。

### CPU 流程验证

安装 pytest 后，可以使用合成样本和小型编码器运行训练与评估流程。

```bash
python -m pytest -q tests/test_train_smoke.py
```
