#!/bin/bash
#SBATCH --job-name=crrl-large
#SBATCH --output=logs/large_%j.out
#SBATCH --error=logs/large_%j.err
#SBATCH --gpus 8
#SBATCH --time=48:00:00
#SBATCH -C "fat"

# Large training job, 8 GPUs, 4 running vLLM, 4 training via accelerate

# Configuration
MODEL_CONFIG="large_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)
GRPO_CONFIG="long"
TP_SIZE=4


# Launch vLLM server on GPUs 4-7
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0,1,2,3 crrl.sif \
    trl vllm-serve --model $MODEL_NAME \
    --tensor_parallel_size 4 &  # & makes it run in the background

# Launch training on GPUs 0-3 with large model configuration using accelerate
apptainer exec --nv --bind "$(pwd):/app" --env CUDA_VISIBLE_DEVICES=4,5,6,7 crrl.sif \
    accelerate launch \
    --config_file scripts/deepspeed_zero3.yaml \
    --num_processes 4 \
    --num_machines 1 \
    /app/src/train_grpo.py \
    model=$MODEL_CONFIG \
    grpo.per_device_train_batch_size=4 \
    grpo.gradient_accumulation_steps=8 \
    grpo.max_prompt_length=2048 \
    grpo.max_completion_length=1024 \
    grpo.num_generations=8 \
    "$@"  # pass any additional arguments
    
# Wait for both processes to finish
wait

# Print end time
echo "End time: $(date)"
