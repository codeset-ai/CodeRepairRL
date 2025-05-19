#!/bin/bash
#SBATCH --job-name=crrl-medium
#SBATCH --output=logs/medium_%j.out
#SBATCH --error=logs/medium_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"

# Medium train job, 2 GPUs, 1 running vLLM, 1 training

# Configuration
MODEL_CONFIG="medium_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

CUDA_VISIBLE_DEVICES=1 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len 8129 \
    --enable-auto-tool-choice \
    --reasoning_parser deepseek_r1 \
    --tool-call-parser hermes \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv crrl.sif \
    python -m src.train_grpo \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo.vllm_mode=async_server \
    grpo.gradient_accumulation_steps=4 \
    grpo.per_device_train_batch_size=1 \
    grpo.num_generations=8 \
    "$@"  # pass any additional arguments
    
wait  # wait for all background processes to finish
