#!/bin/bash
#SBATCH --job-name=crrl-small-sft-lora
#SBATCH --output=logs/small_sft_lora_%j.out
#SBATCH --error=logs/small_sft_lora_%j.err
#SBATCH --gpus 4
#SBATCH --time=24:00:00
#SBATCH -C "fat"

apptainer exec --nv crrl.sif accelerate launch \
    --config_file scripts/deepspeed/zero2.yaml \
    --num_processes 4 \
    --mixed_precision bf16 \
    --module src.train_sft -- \
        model=small_qwen \
        model.lora=false \
        sft.learning_rate=1e-5 \
        "$@"