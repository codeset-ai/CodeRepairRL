#!/bin/bash
#SBATCH --job-name=crrl-medium
#SBATCH --output=logs/medium_%j.out
#SBATCH --error=logs/medium_%j.err
#SBATCH --gpus 3
#SBATCH --time=24:00:00
#SBATCH -C "fat"

# Medium train job, 3 fat GPUs, 1 running vLLM, 2 training

# Model configuration
MODEL_CONFIG="medium_qwen"
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
    accelerate launch src/train_grpo.py \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo=long \
    grpo.gradient_accumulation_steps=4 \
    grpo.per_device_train_batch_size=1 \
    grpo.num_generations=8 \
    grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
    grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
    "$@"  # pass any additional arguments
    
wait  # wait for all background processes to finish
