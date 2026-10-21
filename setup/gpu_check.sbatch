#!/bin/bash
#SBATCH --job-name=rna-env-check
#SBATCH --partition=g-3090-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --output=/home/junxuanliu/projects/RNA_ThermoFusion/logs/gpu-check-%j.out
#SBATCH --error=/home/junxuanliu/projects/RNA_ThermoFusion/logs/gpu-check-%j.err
set -e
source "$HOME/projects/RNA_ThermoFusion/activate.sh"
cd "$RNA_PROJECT_ROOT"
nvidia-smi
srun python setup/check_environment.py --gpu
