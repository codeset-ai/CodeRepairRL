#!/bin/bash
#SBATCH --job-name=crrl-small
#SBATCH --output=logs/small_%j.out
#SBATCH --error=logs/small_%j.err
#SBATCH --gpus 1
#SBATCH --time=08:00:00
#SBATCH -C "fat"

# Small train job, 1 GPU, no vLLM

apptainer exec --nv crrl.sif python -m src.train_grpo \
    model=small_qwen \
    grpo.use_vllm=false \
    grpo.max_prompt_length=1024 \
    grpo.max_completion_length=512 \
    "$@"  # pass any additional arguments