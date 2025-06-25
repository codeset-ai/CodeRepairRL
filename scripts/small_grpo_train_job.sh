#!/bin/bash
#SBATCH --job-name=crrl-small-grpo
#SBATCH --output=logs/small_grpo_%j.out
#SBATCH --error=logs/small_grpo_%j.err
#SBATCH --gpus 5            # 1 serve + 4 train
#SBATCH --time=36:00:00
#SBATCH -C "fat"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Small GRPO train job, 5 fat GPUs, 1 running vLLM, 4 training

# Model configuration - use merged SFT model for simplified VLLM pipeline
MODEL_CONFIG="small_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=7168
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))  # not strictly needed, but so we don't get context window errors

# VLLM server - loads initial model (any same-architecture model works)
# Training server will sync weights from training model before first inference
CUDA_VISIBLE_DEVICES=4 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
        --model "$MODEL_NAME" \
        --max_model_len $VLLM_CONTEXT_LENGTH \
        --gpu-memory-utilization 0.94 \
        --disable-log-stats \
        --enable-auto-tool-choice \
        --reasoning_parser qwen3 \
        --tool-call-parser hermes \
        &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer exec --nv crrl.sif \
accelerate launch \
    --config_file scripts/deepspeed/zero2.yaml \
    --module src.train_grpo -- \
        run=repo_repair \
        model=${MODEL_CONFIG} \
        model.lora=false \
        grpo=multi_turn \
        grpo.num_generations=8 \
        grpo.generation_batch_size=8 \
        grpo.per_device_train_batch_size=2 \
        grpo.gradient_accumulation_steps=1 \
        grpo.max_prompt_length=${MAX_PROMPT_LENGTH} \
        grpo.max_completion_length=${MAX_COMPLETION_LENGTH} \
        "$@"  # pass any additional arguments