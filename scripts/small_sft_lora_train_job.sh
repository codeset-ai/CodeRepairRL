#!/bin/bash
#SBATCH --job-name=crrl-small-sft-lora
#SBATCH --output=logs/small_sft_lora_%j.out
#SBATCH --error=logs/small_sft_lora_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"

apptainer exec --nv crrl.sif accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    --module src.train_sft -- \
        model=small_qwen \
        "$@"
