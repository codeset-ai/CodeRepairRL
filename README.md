# CodeRepairRL - Reinforcement Learning for Program Repair

## Overview

CodeRepairRL leverages recent advancements in applying Reinforcement Learning (RL) to Large Language Models (LLMs) to fine-tune them for domain-specific tasks. Our ultimate goal is to develop models similar to [RepairLLama](https://arxiv.org/pdf/2312.15698) and [Llama-3-SWE-RL](https://arxiv.org/pdf/2502.18449), which "punch above their weight-class" in terms of parameter count, demonstrating exceptional performance in software engineering benchmarks.

For more details on the project's objectives, conceptual background, and implementation specifics, see [docs/PROJECT.md](docs/PROJECT.md).

## Getting Started

### Building the Container

To build the Apptainer container:

```bash
./scripts/build_container.sh
```

This creates a `crrl.sif` file in the project root.

#### Running Training Jobs

We provide specialized SLURM scripts for different model sizes, each pre-configured with appropriate compute resource allocations:

```bash
# For small models (~1.5B range), defaults to Qwen2.5-Coder-1.5B-Instruct
sbatch scripts/small_train_job.sh

# For medium models (~7B range), defaults to Qwen2.5-Coder-7B-Instruct
sbatch scripts/medium_train_job.sh

# For small models (~32B range), defaults to Qwen2.5-Coder-32B-Instruct
sbatch scripts/large_train_job.sh
```

Each script includes pre-tuned GRPO parameters optimized for the corresponding model size category. You can further customize by adding options:

```bash
sbatch scripts/large_train_job.sh --model google/gemma-3-27b-it

# This is already a reasoning model so increasing max_completion_length might be necessary
sbatch scripts/medium_train_job.sh --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
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

To run a training job on a single GPU, we disable vllm. This means that generations take place within the torch environment instead of the highly optimized vllm, making it much slower. A faster (yet unmaintainable) version exists for low resource training utilizing unsloth, that can be found on older commits.
```bash
# Run a script using the virtual environment
uv run -m src.train_grpo grpo.use_vllm=false
```
This is useful for quick testing and development, though for larger scale training the distributed SLURM scripts are recommended.

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
- **docs/PAPERS.md**: Lists relevant research papers and literature reviewed for the project.