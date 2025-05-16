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


apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 crrl.sif \
    trl vllm-serve \ 
    --model $MODEL_NAME \
    # --port 8000 \  # better?
    &  # & makes it run in the background

apptainer exec --nv --env CUDA_VISIBLE_DEVICES=1 crrl.sif \
    python -m src.train_grpo \
    run=repo_repair \
    model=$MODEL_CONFIG \
    grpo.per_device_train_batch_size=8 \
    grpo.gradient_accumulation_steps=4 \
    grpo.max_prompt_length=1024 \
    grpo.max_completion_length=512 \
    grpo.num_generations=8 \
    "$@"  # pass any additional arguments
    
wait  # wait for all background processes to finish
