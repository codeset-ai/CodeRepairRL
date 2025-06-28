#!/bin/bash
#SBATCH --job-name=crrl-small-sft-lora
#SBATCH --output=logs/small_sft_lora_%j.out
#SBATCH --error=logs/small_sft_lora_%j.err
#SBATCH --gpus 1
#SBATCH --time=24:00:00
#SBATCH -C "fat"

apptainer exec --nv crrl.sif \
    python3 -m src.train_sft \
    model=small_qwen \
    "$@"
