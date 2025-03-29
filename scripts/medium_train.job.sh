#!/bin/bash
#SBATCH --job-name=ttc-medium
#SBATCH --output=logs/medium_%j.out
#SBATCH --error=logs/medium_%j.err
#SBATCH --gpus 4
#SBATCH --time=24:00:00
#SBATCH -C "thin"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user bhbj@kth.se

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Directory: $(pwd)"

# Launch vLLM server on GPUs 2-3
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=2,3 ttc.sif \
    trl vllm-serve --model Qwen/Qwen2.5-Coder-7B-Instruct --tensor_parallel_size 2 &

# Wait a moment for vLLM to start
sleep 30

# Launch training on GPUs 0-1 with medium model configuration using accelerate
apptainer exec --nv --bind "$(pwd):/app" --env CUDA_VISIBLE_DEVICES=0,1 ttc.sif \
    accelerate launch \
    --config_file scripts/deepspeed_zero3.yaml \
    --num_processes 2 \
    --num_machines 1 \
    /app/src/train_grpo.py \
    model=medium_qwen \
    grpo=medium \
    "$@"  # pass any additional arguments
    
# Wait for both processes to finish
wait

# Print end time
echo "End time: $(date)"
