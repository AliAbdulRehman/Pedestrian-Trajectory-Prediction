#!/bin/bash
#SBATCH --job-name=LLaVa_Accuracy
#SBATCH --account=project_2009936
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

module --force purge
module load pytorch

export PYTHONUSERBASE=/scratch/project_2009936
export TRANSFORMERS_CACHE=/scratch/project_2009936
export HF_HOME=/scratch/project_2009936

srun python test_accuracy.py

