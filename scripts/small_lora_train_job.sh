#!/bin/bash
#SBATCH --job-name=crrl-small-lora
#SBATCH --output=logs/small_lora_%j.out
#SBATCH --error=logs/small_lora_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Small train job, 2 fat GPUs, 1 running vLLM, 1 training

# Model configuration - use SFT model if available, otherwise base model
MODEL_CONFIG="small_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=7168
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))


CUDA_VISIBLE_DEVICES=1 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len $MAX_CONTEXT_LENGTH \
    --enable-auto-tool-choice \
    --reasoning_parser deepseek_r1 \
    --tool-call-parser hermes \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv crrl.sif \
    python3 -m src.train_grpo \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo=long \
    grpo.gradient_accumulation_steps=1 \
    grpo.per_device_train_batch_size=4 \
    grpo.num_generations=4 \
    grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
    grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
    "$@"  # pass any additional arguments
