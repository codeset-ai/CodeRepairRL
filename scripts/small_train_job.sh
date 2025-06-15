#!/bin/bash
#SBATCH --job-name=crrl-small-full
#SBATCH --output=logs/small_full_%j.out
#SBATCH --error=logs/small_full_%j.err
#SBATCH --gpus 5            # 1 serve + 4 train
#SBATCH --time=36:00:00
#SBATCH -C "fat"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Small train job, 5 fat GPUs, 1 running vLLM, 4 training

# Model configuration - use SFT model if available, otherwise base model
MODEL_CONFIG="small_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=7168
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))


CUDA_VISIBLE_DEVICES=4 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len $MAX_CONTEXT_LENGTH \
    --enable-auto-tool-choice \
    --reasoning_parser deepseek_r1 \
    --tool-call-parser hermes \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer exec --nv crrl.sif \
    python3 -m src.train_grpo \
    run=repo_repair \
    model=$MODEL_CONFIG \
    model.lora=false \
    grpo=long \
    grpo.learning_rate=5e-6 \
    grpo.gradient_accumulation_steps=1 \
    grpo.per_device_train_batch_size=4 \
    grpo.num_generations=4 \
    grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
    grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
    "$@"  # pass any additional arguments
    