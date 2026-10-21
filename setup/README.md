# Mingzhu RNA ThermoFusion environment

Server: 10.22.19.39. Account: junxuanliu.
Project root: /home/junxuanliu/projects/RNA_ThermoFusion
Environment: /home/junxuanliu/projects/RNA_ThermoFusion/.venv

## Activate

```bash
source ~/projects/RNA_ThermoFusion/activate.sh
python --version
which python
```

The environment uses the cluster's Anaconda3/2022.10 activation script but installs Python and all project packages in the private project prefix. No system packages or shared Conda environment are modified. Activation does not change CUDA_VISIBLE_DEVICES; Slurm controls GPU visibility.

## Package choices

Python 3.11.9, PyTorch 2.1.2 + CUDA 11.8, ViennaRNA 2.6.4, NumPy 1.26.4, FlashAttention 2.3.2, RiNALMo 1.0.0. Scientific dependencies are listed in requirements.txt. Full resolved versions are recorded in requirements.lock.txt after installation.

This CentOS 7 server has glibc 2.17. The old project's PyTorch 2.8/CUDA 12.8 environment cannot simply be copied here. The chosen Torch 2.1 series also matches the original RiNALMo experiment environment. NumPy remains below 2 for compatibility. ViennaRNA is pinned to the version required by the project's thermodynamic cache implementation. Its C stack-probability symbol is tested explicitly.

Official references:
- https://pytorch.org/get-started/previous-versions/
- https://github.com/lbcb-sci/RiNALMo
- https://github.com/Dao-AILab/flash-attention/releases/tag/v2.3.2
- https://pypi.org/project/ViennaRNA/2.6.4/

## CPU and import checks

```bash
source ~/projects/RNA_ThermoFusion/activate.sh
python "$RNA_PROJECT_ROOT/setup/check_environment.py"
python -m pip check
```

The check uses tiny synthetic inputs. It checks imports, CPU autograd, the NumPy/Torch bridge, RiNALMo configuration and tokenization, ViennaRNA BPP and the real stack-probability C API. The JSON report is environment_validation.json. These checks do not test GPU execution.

## GPU validation through Slurm

The login node master has no visible NVIDIA GPU. The Slurm GPU partition is g-3090-2, containing gpu0001 through gpu0004, each advertising two GPUs. The account/QOS is junxuanliu/normal.

The environment validation job 9115 completed successfully on gpu0002 with one NVIDIA GeForce RTX 3090. The template requests one GPU, two CPU threads, 8 GiB RAM and three minutes. To rerun it:

```bash
sbatch ~/projects/RNA_ThermoFusion/setup/gpu_check.sbatch
squeue -u junxuanliu
```

The job checks nvidia-smi, BF16 and FlashAttention on the allocated GPU. Output goes to logs/gpu-check-JOBID.out and .err; a successful check writes setup/gpu_validation.json. The recorded GPU, BF16 and FlashAttention checks all passed.

## Scope

These scripts prepare the runtime and RiNALMo library. The full project reproduction is a separate stage, documented in the repository README and run with scripts/reproduce_cluster.py.

## Installation records

- conda_create.log: logs/conda_create.log
- scientific packages: logs/install_runtime_retry.log
- official Torch wheel download and SHA256: logs/download_torch.log and setup/torch_wheel_sha256.txt
- remaining installation: logs/finish_install_resume.log
- detailed import check: logs/environment_validation.log
- RiNALMo source revision: setup/rinalmo_commit.txt

## Verified result

Installation completed on 2026-10-21. pip check reports no broken requirements. All 21 package imports, CPU backward propagation, NumPy/Torch interoperability, TensorBoard event writing, RiNALMo configuration/tokenization, ViennaRNA BPP and the C stack-probability API passed. GPU validation job 9115 also passed.
