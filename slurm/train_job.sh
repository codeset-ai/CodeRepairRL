#!/bin/bash
#SBATCH -A Berzelius-2024-336
#SBATCH --job-name=ttc-test
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gpus 2
#SBATCH --time=0:10:00
#SBATCH -C fat
#SBATCH --mail-type=FAIL
#SBATCH --mail-user bhbj@kth.se

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment
module load buildenv-gcccuda/12.1.1-gcc12.3.0

# Set up uv for dependency management
if ! command -v uv &> /dev/null; then
    echo "Downloading uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"


# Default configuration
SCRIPT="src/train_grpo.py"
CONFIG="base_grpo_config"
NUM_GPUS=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --unsloth)
      SCRIPT="src/train_grpo_unsloth.py"
      shift
      ;;
    --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the training script with torchrun for distributed training
echo "Running: torchrun --nproc-per-node=$NUM_GPUS $SCRIPT +experiment=$CONFIG"
uv run torchrun --nproc-per-node=$NUM_GPUS $SCRIPT +experiment=$CONFIG


# Optional: copy important data from scratch to your project directory
# For example:
cp -r /scratch/local/$SLURM_JOB_ID/results /proj/berzelius-2025-72/users/x_bjabj/

# Print end time
echo "End time: $(date)" 
