#!/bin/bash
#SBATCH --job-name=crrl-small-sft-lora
#SBATCH --output=logs/small_sft_lora_%j.out
#SBATCH --error=logs/small_sft_lora_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"

# Small SFT train job, 2 fat GPUs
CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv crrl.sif \
    accelerate launch \
    --config_file scripts/deepspeed/zero2.yaml \
    src/train_sft.py \
    model=small_qwen \
    "$@"