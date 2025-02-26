#!/bin/bash
#SBATCH -A Berzelius-2024-336
#SBATCH --job-name=ttc-test
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

# Create logs directory if it doesn't exist
mkdir -p logs


# Define container variables
CONTAINER_IMAGE="$(pwd)/ttc.sif"
PROJECT_DIR="$(pwd)"
SCRATCH_DIR="/home/x_bjabj/scratch/$SLURM_JOB_ID"

# Create scratch directory
mkdir -p $SCRATCH_DIR

# Check if container exists
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "Error: Container image not found at $CONTAINER_IMAGE"
    exit 1
fi

# Print Python and CUDA information
echo "Python version:"
apptainer exec --nv $CONTAINER_IMAGE python --version

echo "CUDA availability:"
apptainer exec --nv $CONTAINER_IMAGE python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
apptainer exec --nv $CONTAINER_IMAGE python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
apptainer exec --nv $CONTAINER_IMAGE python -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

# Test importing key libraries
echo "Testing imports..."
apptainer exec --nv $CONTAINER_IMAGE python -c "
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
apptainer exec --nv $CONTAINER_IMAGE torchrun --nproc-per-node=1 -m torch.distributed.run --help

# Print end time
echo "End time: $(date)" 