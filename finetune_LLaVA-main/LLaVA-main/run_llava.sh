#!/bin/bash
#SBATCH --job-name=LLaVa_Run
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

srun python -m llava.eval.run_llava \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --model-path  "checkpoints1/llava-v1.5-7b-task-lora" \
    --image-file "/scratch/project_2009936/LLaVA-main/playground/data/images/video_0243/00059.png" \
    --query "Perception and Prediction: -Past History:-[] -Pedestrian Attributes:- Age:adult -Gender:female -Group Size:1 person(s) -Pedestrian Actions:- Reaction:undefined -Hand Gesture:undefined -Look:not-looking -Nodding:not-nodding -Scene Information: -Designated Crosswalk:no -Motion Direction:latitudinal -Number of Lanes:Unknown -Signalized Intersection:no information -Traffic Direction:one_way -Pedestrian Crossing:no -Pedestrian Sign:no -Stop Sign:no -Traffic Light:no information -Road Type:street -Mission Goal: Predict if the pedestrian is going to cross the road or not?"

