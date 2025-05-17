#!/bin/bash
#SBATCH --job-name=crrl-small
#SBATCH --output=logs/small_%j.out
#SBATCH --error=logs/small_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "thin"

# Small train job, 2 thin GPUs, 1 running vLLM, 1 training

# Configuration
MODEL_CONFIG="small_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=1 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len 32768 \
    --enable-auto-tool-choice \
    --reasoning_parser deepseek_r1 \
    --tool-call-parser hermes \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 apptainer exec --nv crrl.sif \
    python -m src.train_grpo \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo.vllm_mode=async_server \
    grpo.per_device_train_batch_size=2 \
    grpo.gradient_accumulation_steps=4 \
    grpo.num_generations=4 \
    "$@"  # pass any additional arguments
    
wait  # wait for all background processes to finish
