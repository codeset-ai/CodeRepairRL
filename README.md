# CodeRepairRL - Reinforcement Learning for Program Repair

## Overview

CodeRepairRL leverages recent advancements in applying Reinforcement Learning (RL) to Large Language Models (LLMs) to fine-tune them for domain-specific tasks. Our ultimate goal is to develop models similar to [RepairLLama](https://arxiv.org/pdf/2312.15698) and [Llama-3-SWE-RL](https://arxiv.org/pdf/2502.18449), which "punch above their weight-class" in terms of parameter count, demonstrating exceptional performance in software engineering benchmarks.

The project uses a two-stage training approach:
1. **Supervised Fine-Tuning (SFT)**: Initial fine-tuning on high-quality code repair demonstrations
2. **Group Relative Policy Optimization (GRPO)**: Reinforcement learning to further improve performance on specific tasks

For more details on the project's objectives, conceptual background, and implementation specifics, see [docs/PROJECT.md](docs/PROJECT.md).

## Academic Paper

The methodology and findings of this project are documented in an academic paper. The LaTeX repository for the paper is available at [CodeRepairRL-Paper](https://github.com/BjarniHaukur/CodeRepairRL-Paper).

## Getting Started

### Building the Container

To build the Apptainer container:

```bash
# Build the training container 
apptainer build crrl.sif scripts/train_container.def
```

(the build process may take several minutes)

### Running Supervised Fine-Tuning (SFT)

Before GRPO training, you can optionally run SFT to create a better starting point:

```bash
# Run SFT training job
sbatch scripts/sft_train_job.sh

# Or run locally for testing
uv run -m src.train_sft
```

The SFT stage uses curated datasets of high-quality code repair examples to provide the model with a strong foundation before RL training.

### Running GRPO Training Jobs

We provide specialized SLURM scripts for different model sizes, each pre-configured with appropriate compute resource allocations:

```bash
# For small models (8B), defaults to Qwen/Qwen3-8B
sbatch scripts/small_train_job.sh

# For medium models (8B with higher LoRA rank), defaults to Qwen/Qwen3-8B
sbatch scripts/medium_train_job.sh

# For large models (32B), defaults to Qwen/Qwen3-32B
sbatch scripts/large_train_job.sh
```

Each script includes pre-tuned GRPO parameters optimized for the corresponding model size category. The scripts support three task types:
- **detection**: Binary vulnerability detection
- **repair**: Single-file code repair with search-replace diffs
- **repo_repair**: Repository-level code repair using agentic approaches

You can customize training with Hydra overrides:

```bash
# Change task type
sbatch scripts/medium_train_job.sh run.task_type=detection

# Use a different model
sbatch scripts/large_train_job.sh model.model_name=meta-llama/Llama-3.1-70B-Instruct

# Start from an SFT checkpoint
sbatch scripts/medium_train_job.sh model.lora_checkpoint_path=/path/to/sft/checkpoint
```

## Local Development

For "local" development and testing without Apptainer containers, you can use `uv` directly.

### Installing uv

Install the `uv` package manager with:

MacOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (project not tested on Windows)
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Optimizing Cache Locations

It is recommended to move cache locations to your project directory by adding the following to your `.bashrc` or `.zshrc`:

```bash
# Define your project directory
export PROJECT_DIR=/proj/berzelius-2024-336/users/x_bjabj/

# Set Hugging Face cache locations
export HF_HOME=$PROJECT_DIR/.hf
export TRANSFORMERS_CACHE=$PROJECT_DIR/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$PROJECT_DIR/.cache/huggingface/datasets

# Set uv cache location
export UV_CACHE_DIR=$PROJECT_DIR/.cache/.uv
```

This ensures that large model files and datasets are stored in your project directory, which has way more storage space than your home directory.

### Setting Up Development Environment

`uv` will automatically recognize the Python version specified in `.python-version` and set up a virtual environment accordingly:

For local development and testing:

```bash
# Run GRPO training without vLLM (slower but works on single GPU)
uv run -m src.train_grpo grpo.use_vllm=false

# Run SFT training
uv run -m src.train_sft

# Curate SFT datasets
uv run -m src.curate_sft_data
```

Note: Disabling vLLM means generations happen in the PyTorch environment instead of the highly optimized vLLM server, making it much slower. For production training, use the SLURM scripts with vLLM enabled.

### Testing

```bash
# run all tests
uv run pytest

# run specific testing file
uv run pytest tests/test_search_replace_diff.py

# run specific test
uv run pytest tests/test_search_replace_diff.py::test_specific_function
```

## Documentation Structure

This repository uses several Markdown files to organize information:

- **README.md**: (This file) Provides a high-level overview, setup instructions, and basic usage examples.
- **docs/PROJECT.md**: Contains detailed information about the project's goals, implementation notes, theoretical background, and conceptual insights.
- **docs/DIARY.md**: A development diary tracking progress, challenges, and decisions.
- **docs/AGENT_RL_INTEGRATION.md**: Describes our approach to integrating agent frameworks into RL training loops using OpenAI-compatible API servers.
- **docs/DATASETS.md**: Describes the datasets used in the project.
- **docs/RESOURCES.md**: Lists relevant research papers, literature and broader resources reviewed for the project.
- **docs/VOCABULARY.md**: Defines key terms and concepts used throughout the project.
- **docs/PAPER.md**: Outlines the structure and key points for the academic paper.