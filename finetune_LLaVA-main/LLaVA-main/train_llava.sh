#!/bin/bash
#SBATCH --job-name=LLava_Train
#SBATCH --account=project_2009936
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

module --force purge
module load pytorch

export PYTHONUSERBASE=/scratch/project_2009936
export TRANSFORMERS_CACHE=/scratch/project_2009936
export HF_HOME=/scratch/project_2009936

pip install --user deepspeed

srun singularity_wrapper exec deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version ped1 \
    --data_path ./playground/data/overall_train_prompts_aa.json \
    --validation_data_path ./playground/data/overall_val_prompts_aa.json \
    --image_folder ./playground/data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints1/llava-v1.5-7b-task-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

