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

apptainer exec --nv ttc.sif python src/train_grpo.py --config-name="grpo_config_large" grpo.run_name="deepseek/Qwen2.5-Coder-7B-Vuln-Classification-2"

# Print end time
echo "End time: $(date)" 
