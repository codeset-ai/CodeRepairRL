#!/bin/bash
#SBATCH --job-name=swe-gym-sft
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"

# SFT training with DeepSpeed on curated SWE-Gym data
# Uses 2 GPUs for training

# Note: HuggingFace authentication is handled by whoami() check in train_sft.py
# Make sure you've run 'huggingface-cli login' before submitting this job

CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv crrl.sif \
    accelerate launch \
    --config_file scripts/deepspeed/zero2.yaml \
    src/train_sft.py \
    "$@"  # Pass any Hydra overrides