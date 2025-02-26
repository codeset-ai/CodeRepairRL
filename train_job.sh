#!/bin/bash
#SBATCH --job-name=ttc
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH -A berzelius-2025-72         # Specify your Berzelius project account here
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1            # Request 1 A100 GPU. (Update manually if using >1 GPU.)
#SBATCH --mem=128G                 # Adjust memory to match the default for a 1-GPU job
#SBATCH --time=0:10:00
#SBATCH --partition=gpu            # Ensure this is the correct Berzelius GPU partition
#SBATCH -D /proj/berzelius-2025-72/users/x_bjabj   # Set working directory (optional)
#SBATCH -C fat
#SBATCH --mail-type=FAIL           # Optional: get notified on failure
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
NUM_GPUS=1

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
echo "Running: torchrun --nproc_per_node=$NUM_GPUS $SCRIPT +experiment=$CONFIG"
uv run torchrun --nproc_per_node=$NUM_GPUS $SCRIPT +experiment=$CONFIG


# Optional: copy important data from scratch to your project directory
# For example:
cp -r /scratch/local/$SLURM_JOB_ID/results /proj/berzelius-2025-72/users/x_bjabj/

# Print end time
echo "End time: $(date)" 
