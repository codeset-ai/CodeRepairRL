#!/bin/bash
#SBATCH --job-name=crrl-test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --gpus 2
#SBATCH --time=0:02:00
#SBATCH -C "thin"

echo "CUDA availability:"
apptainer exec --nv crrl.sif python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
apptainer exec --nv crrl.sif python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
apptainer exec --nv crrl.sif python -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"