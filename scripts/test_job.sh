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

# Set up environment
module load buildenv-gcccuda/12.1.1-gcc12.3.0
module load apptainer

# Define container variables
CONTAINER_IMAGE="ttc.sif"
PROJECT_DIR="$(pwd)"
SCRATCH_DIR="/scratch/local/$SLURM_JOB_ID"

# Create scratch directory
mkdir -p $SCRATCH_DIR

# Define bind paths
BIND_PATHS="$PROJECT_DIR,$SCRATCH_DIR"
for dir in data models; do
  if [ -d "$PROJECT_DIR/$dir" ]; then
    BIND_PATHS="$BIND_PATHS,$PROJECT_DIR/$dir:/opt/ttc/$dir"
  fi
done

# Print Python and CUDA information
echo "Python version:"
apptainer run --nv --bind $BIND_PATHS $CONTAINER_IMAGE python --version

echo "CUDA availability:"
apptainer run --nv --bind $BIND_PATHS $CONTAINER_IMAGE python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
apptainer run --nv --bind $BIND_PATHS $CONTAINER_IMAGE python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
apptainer run --nv --bind $BIND_PATHS $CONTAINER_IMAGE python -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

# Test importing key libraries
echo "Testing imports..."
apptainer run --nv --bind $BIND_PATHS $CONTAINER_IMAGE python -c "
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
apptainer run --nv --bind $BIND_PATHS $CONTAINER_IMAGE torchrun --nproc-per-node=1 -m torch.distributed.run --help

# Print end time
echo "End time: $(date)" 