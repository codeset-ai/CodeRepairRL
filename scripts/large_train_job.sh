#!/bin/bash
#SBATCH --job-name=ttc-large
#SBATCH --output=logs/large_%j.out
#SBATCH --error=logs/large_%j.err
#SBATCH --gpus 8
#SBATCH --time=48:00:00
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

# Launch vLLM server on GPUs 4-7
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=4,5,6,7 ttc.sif \
    trl vllm-serve --model Qwen/Qwen2.5-Coder-32B-Instruct --tensor_parallel_size 4 &

# Wait a moment for vLLM to start
sleep 30

# Launch training on GPUs 0-3 with large model configuration using accelerate
apptainer exec --nv --bind "$(pwd):/app" --env CUDA_VISIBLE_DEVICES=0,1,2,3 ttc.sif \
    accelerate launch \
    --config_file scripts/deepspeed_zero3.yaml \
    --num_processes 4 \
    --num_machines 1 \
    /app/src/train_grpo.py \
    model=large_qwen \
    grpo=long \
    "$@"  # pass any additional arguments
    
# Wait for both processes to finish
wait

# Print end time
echo "End time: $(date)"
