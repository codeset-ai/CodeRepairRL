#!/bin/bash
#SBATCH --job-name=ttc-medium
#SBATCH --output=logs/medium_%j.out
#SBATCH --error=logs/medium_%j.err
#SBATCH --gpus 4
#SBATCH --time=24:00:00
#SBATCH -C "thin"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user bhbj@kth.se

# Configuration
MODEL_CONFIG="medium_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)
GRPO_CONFIG="medium"
TP_SIZE=2

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Directory: $(pwd)"
echo "Using model: $MODEL_NAME"

# Launch vLLM server on GPUs 2-3
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=2,3 ttc.sif \
    trl vllm-serve --model $MODEL_NAME --tensor_parallel_size $TP_SIZE &

# Launch training on GPUs 0-1 with medium model configuration using accelerate
apptainer exec --nv --bind "$(pwd):/app" --env CUDA_VISIBLE_DEVICES=0,1 ttc.sif \
    accelerate launch \
    --config_file scripts/deepspeed_zero3.yaml \
    --num_processes 2 \
    --num_machines 1 \
    /app/src/train_grpo.py \
    model=$MODEL_CONFIG \
    grpo=$GRPO_CONFIG \
    "$@"  # pass any additional arguments
    
# Wait for both processes to finish
wait

# Print end time
echo "End time: $(date)"
