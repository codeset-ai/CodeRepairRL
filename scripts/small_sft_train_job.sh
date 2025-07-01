#!/bin/bash
#SBATCH --job-name=crrl-small-sft-full
#SBATCH --output=logs/small_sft_full_%j.out
#SBATCH --error=logs/small_sft_full_%j.err
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
        sft.max_length=12288 \
        sft.per_device_train_batch_size=2 \
        sft.gradient_accumulation_steps=8 \
        sft.packing=false \
        sft.ddp_bucket_cap_mb=16 \
        sft.ddp_find_unused_parameters=false \
        "$@"