# TTC - Test Time Compute for Program Repair

## Project Structure

- `src/`: Source code for the project
- `scripts/`: Scripts for building and running the project
  - `container.def`: Apptainer container definition
  - `build_container.sh`: Script to build the container
  - `train_job.sh`: SLURM script for training
  - `test_job.sh`: SLURM script for testing

## Getting Started

### Building the Container

To build the Apptainer container:

```bash
./scripts/build_container.sh
```

This will create a `ttc.sif` file in the project root.

### Running Training Jobs

To submit a training job to SLURM:

```bash
sbatch scripts/train_job.sh --config your_config
```

Options:
- `--config CONFIG`: Specify the configuration to use
- `--unsloth`: Use the unsloth training script
- `--gpus N`: Specify the number of GPUs to use

### Running Test Jobs

To submit a test job to SLURM:

```bash
sbatch scripts/test_job.sh
```

## Container Details

The container uses uv for dependency management and includes all necessary Python packages specified in the pyproject.toml file. It's built on Python 3.11 and includes CUDA support for GPU training.

## GPU Precision Auto-Detection

The training configuration automatically detects the GPU architecture and sets the appropriate precision settings:

- For Ampere (SM 8.0) and newer GPUs (e.g., A100, A6000, RTX 3000/4000 series), BF16 precision will be used
- For Pascal (SM 6.0) to Turing (SM 7.5) GPUs (e.g., GTX 1000 series, RTX 2000 series), FP16 precision will be used

