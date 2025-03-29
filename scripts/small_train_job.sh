#!/bin/bash
#SBATCH --job-name=ttc-small
#SBATCH --output=logs/small_%j.out
#SBATCH --error=logs/small_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "thin"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user bhbj@kth.se

# Configuration
MODEL_CONFIG="small_qwen"
MODEL_NAME=$(grep -Po 'model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)
GRPO_CONFIG="short"
TP_SIZE=1

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Directory: $(pwd)"
echo "Using model: $MODEL_NAME"


# Launch vLLM server on GPU 1
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=1 ttc.sif \
    trl vllm-serve --model $MODEL_NAME --tensor_parallel_size $TP_SIZE &  # ampere's and makes it run in background 

# Launch training on GPU 0 with small model configuration using accelerate
apptainer exec --nv --bind "$(pwd):/app" --env CUDA_VISIBLE_DEVICES=0 ttc.sif \
    accelerate launch \
    --config_file scripts/deepspeed_zero3.yaml \
    --num_processes 1 \
    --num_machines 1 \
    /app/src/train_grpo.py \
    model=$MODEL_CONFIG \
    grpo=$GRPO_CONFIG \
    "$@"  # pass any additional arguments
    
# Wait for both processes to finish
wait

# Print end time
echo "End time: $(date)" 
