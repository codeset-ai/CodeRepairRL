#!/bin/bash
#SBATCH --job-name=crrl-large
#SBATCH --output=logs/large_%j.out
#SBATCH --error=logs/large_%j.err
#SBATCH --gpus 4
#SBATCH --time=48:00:00
#SBATCH -C "fat"

# Large training job, 4 GPUs, 2 running vLLM, 2 training

# Configuration
MODEL_CONFIG="large_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)
TP_SIZE=2


CUDA_VISIBLE_DEVICES=2,3 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len 8192 \
    --enable-auto-tool-choice \
    --reasoning_parser deepseek_r1 \
    --tool-call-parser hermes \
    --tensor_parallel_size $TP_SIZE \
    &  # & makes it run in the background

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv crrl.sif \
    python -m src.train_grpo \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo.vllm_mode=async_server \
    grpo.gradient_accumulation_steps=4 \
    grpo.per_device_train_batch_size=1 \
    grpo.num_generations=8 \
    "$@"  # pass any additional arguments
    
wait  # wait for all background processes to finish