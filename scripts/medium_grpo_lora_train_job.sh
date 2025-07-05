#!/bin/bash
#SBATCH --job-name=crrl-medium-grpo
#SBATCH --output=logs/medium_grpo_%j.out
#SBATCH --error=logs/medium_grpo_%j.err
#SBATCH --gpus 6
#SBATCH --time=48:00:00
#SBATCH -C "fat"

# Large GRPO training job, 4 GPUs, 2 running vLLM, 2 training

# Model configuration - use merged SFT model for simplified VLLM pipeline
MODEL_CONFIG="medium_qwen"
MODEL_NAME=$(grep -Po '^model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

# Context window configuration, this defines our compute requirements more than anything else
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=2048
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))  # not strictly needed, but so we don't get context window errors

# VLLM server - loads initial model (any same-architecture model works)
# Training server will sync weights from training model before first inference
CUDA_VISIBLE_DEVICES=4,5 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len $VLLM_CONTEXT_LENGTH \
    --disable-log-stats \
    --enable-auto-tool-choice \
    --reasoning_parser qwen3 \
    --tool-call-parser hermes \
    --tensor_parallel_size 2 \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
# Overrides the gradient clipping in the config file
CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer exec --nv crrl.sif accelerate launch \
    --config_file scripts/deepspeed/zero3_grpo.yaml \
    --num_processes 4 \
    --module src.train_grpo -- \
        run=repo_repair \
        model=$MODEL_CONFIG \
        grpo=multi_turn \
        grpo.gradient_accumulation_steps=4 \
        grpo.per_device_train_batch_size=2 \
        grpo.num_generations=8 \
        grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
        grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
        grpo.optim="adamw_torch" \
        grpo.ddp_bucket_cap_mb=16 \
        grpo.ddp_find_unused_parameters=false \
        "$@"  # pass any additional arguments
    