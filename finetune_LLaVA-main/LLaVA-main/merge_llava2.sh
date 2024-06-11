#!/bin/bash
#SBATCH --job-name=LLaVa_Merge
#SBATCH --account=project_2009936
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1

module --force purge
module load pytorch

export PYTHONUSERBASE=/scratch/project_2009936
export TRANSFORMERS_CACHE=/scratch/project_2009936
export HF_HOME=/scratch/project_2009936

srun python scripts/merge_lora_weights.py \
    --model-path "checkpoints1/llava-v1.5-7b-task-lora" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --save-model-path "merged_checkpoints1/llava-v1.5-13b-task-lora"

