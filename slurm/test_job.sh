#!/bin/bash
#SBATCH -A berzelius-2025-72
#SBATCH --job-name=ttc
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2            
#SBATCH --mem=256G
#SBATCH --time=0:10:00
#SBATCH --partition=hopper-prod            # is this correct?
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

# Print Python and CUDA information
echo "Python version:"
uv run python --version

echo "CUDA availability:"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
uv run python -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

# Test importing key libraries
echo "Testing imports..."
uv run python -c "
import torch
import transformers
import trl
import wandb
import hydra
import datasets
print('All imports successful!')
"

# Test torchrun
echo "Testing torchrun..."
uv run torchrun --nproc-per-node=1 -m torch.distributed.run --help

# Print end time
echo "End time: $(date)" 