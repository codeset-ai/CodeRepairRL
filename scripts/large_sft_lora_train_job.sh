#!/bin/bash
#SBATCH --job-name=crrl-large-sft-lora
#SBATCH --output=logs/large_sft_lora_%j.out
#SBATCH --error=logs/large_sft_lora_%j.err
#SBATCH --gpus 4
#SBATCH --time=72:00:00
#SBATCH -C "fat"

# Large SFT train job, 4 fat GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer exec --nv crrl.sif \
    accelerate launch \
    --config_file scripts/deepspeed/zero2.yaml \
    src/train_sft.py \
    model=large_qwen \
    "$@"