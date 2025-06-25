#!/bin/bash
#SBATCH --job-name=crrl-large-grpo
#SBATCH --output=logs/large_grpo_%j.out
#SBATCH --error=logs/large_grpo_%j.err
#SBATCH --gpus 4
#SBATCH --time=48:00:00
#SBATCH -C "fat"

# Large GRPO training job, 4 GPUs, 2 running vLLM, 2 training

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Model configuration - use merged SFT model for simplified VLLM pipeline
MODEL_CONFIG="large_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)
TP_SIZE=2

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=7168
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))  # not strictly needed, but so we don't get context window errors

# VLLM server - loads initial model (any same-architecture model works)
# Training server will sync weights from training model before first inference
CUDA_VISIBLE_DEVICES=2,3 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len $VLLM_CONTEXT_LENGTH \
    --gpu-memory-utilization 0.94 \
    --disable-log-stats \
    --enable-auto-tool-choice \
    --reasoning_parser qwen3 \
    --tool-call-parser hermes \
    --tensor_parallel_size $TP_SIZE \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv crrl.sif \
    python -m src.train_grpo \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo=multi_turn \
    grpo.gradient_accumulation_steps=4 \
    grpo.per_device_train_batch_size=1 \
    grpo.num_generations=8 \
    grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
    grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
    "$@"  # pass any additional arguments
    