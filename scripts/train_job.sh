#!/bin/bash
#SBATCH --job-name=ttc-train
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gpus 2
#SBATCH --time=0:10:00
#SBATCH -C "thin"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user bhbj@kth.se

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Directory: $(pwd)"

apptainer exec --nv --bind "$(pwd):/app" ttc.sif python /app/src/train_grpo.py grpo.run_name="deepseek/Qwen2.5-Coder-1.5B-Vuln-Classification-2"

# Print end time
echo "End time: $(date)" 
