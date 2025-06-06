#!/bin/bash
#SBATCH --job-name=crrl-small
#SBATCH --output=logs/small_%j.out
#SBATCH --error=logs/small_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"

# Small train job, 2 fat GPUs, 1 running vLLM, 1 training

# Model configuration - use SFT model if available, otherwise base model
MODEL_CONFIG="small_qwen"
SFT_MODEL_NAME="bjarni/qwen3-8b-swe-gym-sft"
BASE_MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

# Try to use SFT model, fall back to base model if not available
MODEL_NAME="$SFT_MODEL_NAME"

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=7168
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))

# Install flash-attn with GPU support
apptainer run --nv crrl.sif install-flash-attn

CUDA_VISIBLE_DEVICES=1 apptainer run --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len $MAX_CONTEXT_LENGTH \
    --enable-auto-tool-choice \
    --reasoning_parser deepseek_r1 \
    --tool-call-parser hermes \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0 apptainer run --nv crrl.sif \
    accelerate launch src/train_grpo.py \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo=long \
    grpo.gradient_accumulation_steps=1 \
    grpo.per_device_train_batch_size=6 \
    grpo.num_generations=6 \
    grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
    grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
    "$@"  # pass any additional arguments
    
wait  # wait for all background processes to finish
