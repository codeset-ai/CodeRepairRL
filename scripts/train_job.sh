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

# Default configuration
SCRIPT="src/train_grpo.py"
CONFIG="base_grpo_config"
NUM_GPUS=2
CONTAINER_IMAGE="ttc.sif"
PROJECT_DIR="$(pwd)"
SCRATCH_DIR="/proj/berzelius-2024-336/users/x_bjabj/$SLURM_JOB_ID"

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

# Create scratch directory
mkdir -p $SCRATCH_DIR

# Define bind paths
BIND_PATHS="$PROJECT_DIR,$SCRATCH_DIR"
for dir in data models; do
  if [ -d "$PROJECT_DIR/$dir" ]; then
    BIND_PATHS="$BIND_PATHS,$PROJECT_DIR/$dir:/opt/ttc/$dir"
  fi
done

# Run the training script with Apptainer
echo "Running: torchrun --nproc-per-node=$NUM_GPUS $SCRIPT +experiment=$CONFIG"
apptainer run --nv --bind $BIND_PATHS $CONTAINER_IMAGE torchrun --nproc-per-node=$NUM_GPUS $SCRIPT +experiment=$CONFIG

# Copy results from scratch to project directory if needed
if [ -d "$SCRATCH_DIR/results" ]; then
  echo "Copying results from scratch directory..."
  cp -r $SCRATCH_DIR/results /proj/berzelius-2025-72/users/x_bjabj/
fi

# Print end time
echo "End time: $(date)" 
