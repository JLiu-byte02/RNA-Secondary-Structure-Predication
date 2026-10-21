#!/bin/bash
# Source this script: source ~/projects/RNA_ThermoFusion/activate.sh
RNA_PROJECT_ROOT="$HOME/projects/RNA_ThermoFusion"
source /public/software/.local/easybuild/software/Anaconda3/2022.10/etc/profile.d/conda.sh
conda activate "$RNA_PROJECT_ROOT/.venv" || return
export RNA_PROJECT_ROOT PYTHONNOUSERSITE=1
# Triton invokes ldconfig while compiling rotary kernels on GPU nodes.
export PATH="$PATH:/usr/sbin:/sbin"
export TRITON_CACHE_DIR="$RNA_PROJECT_ROOT/.cache/triton"
export TMPDIR="$RNA_PROJECT_ROOT/.tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PIP_CACHE_DIR="$RNA_PROJECT_ROOT/.cache/pip"
export HF_HOME="$RNA_PROJECT_ROOT/.cache/huggingface" TORCH_HOME="$RNA_PROJECT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$RNA_PROJECT_ROOT/.cache" MPLCONFIGDIR="$RNA_PROJECT_ROOT/.cache/matplotlib"
# CUDA_VISIBLE_DEVICES is set by Slurm; preserve it.
