# TTC Project Guidelines

## Commands
- Training: `python src/train_grpo.py`
- Training with unsloth: `python src/train_grpo_unsloth.py`
- Training with config: `python src/train_grpo.py --config grpo_config_large`
- Building container: `./scripts/build_container.sh`
- SLURM job: `sbatch scripts/train_job.sh --config your_config [--unsloth] [--gpus N]`

## Code Style
- **Imports**: Standard imports first, then third-party, then project modules
- **Types**: Use type hints for all function parameters and return values
- **Naming**: Snake_case for variables and functions, PascalCase for classes
- **Config**: Use Hydra for configuration management
- **Logging**: Use wandb for experiment tracking
- **GPU Utils**: Use utils.gpu_utils for automatic precision detection

## Project Structure
- `src/` - Source code (training, evaluation, datasets)
- `data/` - Dataset files (PrimeVul)
- `scripts/` - Build and job scripts
- `conf/` - Configuration files