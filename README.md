# CodeRepairRL - Reinforcement Learning for Program Repair

## Overview

CodeRepairRL leverages recent advancements in applying Reinforcement Learning (RL) to Large Language Models (LLMs) to fine-tune them for domain-specific tasks. Our ultimate goal is to develop models similar to [RepairLLama](https://arxiv.org/pdf/2312.15698) and [Llama-3-SWE-RL](https://arxiv.org/pdf/2502.18449), which "punch above their weight-class" in terms of parameter count, demonstrating exceptional performance in software engineering benchmarks.

## Implementation Note

This project uses a [custom fork of TRL](https://github.com/BjarniHaukur/trl) (Transformer Reinforcement Learning), currently enhanced for better Weights & Biases logging. The fork will be essential for future research directions, e.g. integrating coding agents directly into the generation loop and rewarding correctness end-to-end.

## Objectives

### Code Repair

Our primary objective is generating accurate code patches to fix specific software issues. Initially, we reward the model based on the accuracy of the generated diffs using a search/replace approach that compares model-generated diffs against oracle solutions. This approach offers flexibility for potential future multi-file edits while maintaining simplicity. Ultimately, we aim to use software tests as the oracle for correctness, though this presents challenges due to the extensive compute requirements for running comprehensive test suites.

### Code Implementation from Specification

We also explore the task of implementing code from specifications using our stack.py dataset. This represents a related but distinct challenge from code repair, with fewer existing datasets in this category. The model must generate a complete implementation based on specifications, testing its understanding of requirements, adhering to contextual style and its ability to produce functional code.

### Vulnerability Classification

As a secondary objective, we address classification of software vulnerabilities. This task evaluates our model's ability to reason about complex software vulnerabilities, effectively creating a test-time compute-enabled, process-verifiable classifier. Unlike traditional black-box models, our approach provides explicit reasoning paths, enhancing transparency and trustworthiness.

For detailed run results and progress on our objectives, see our [WandB project page](https://wandb.ai/assert-kth/TTC).

## Post-Training Perspective

Post-training of LLMs is a complex, multivariate process consisting of many specialized steps across various domains, ultimately resulting in powerful general-purpose models. While achieving a comprehensive post-training pipeline is beyond the scope of a master's thesis, our project aims to demonstrate a significant performance improvement in the specific domain of code repair. Such targeted improvements could represent one valuable step in the broader post-training stage of frontier models.

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

## Compute Efficiency

Since LLMs are inherently large, training them can be challenging for individual researchers. Our project previously included Unsloth optimizations (available in older commits), but we have since moved to a more streamlined approach using DeepSpeed and Accelerate for distributed training. We use "ZeRO-Stage 3" DeepSpeed configuration, where optimizer state, gradients, and model parameters are all sharded between GPUs.

## Definitions and Intuitions

In our project, we use specific terms that are crucial for understanding our approach:

- **KL-divergence (Kullback-Leibler divergence)**: A fundamental metric in information theory that measures how one probability distribution differs from another. In our reinforcement learning setup, we use it to quantify how much our updated policy ($\pi_{t+1}$) diverges from our reference policy ($\pi_t$) as the training progresses. This helps ensure our model learns gradually and stably.

- **Loss**: In the context of our model, 'loss' refers to the discrepancy between the reference policy and the update policy. It is a measure of how much the model's predictions deviate from the expected outcomes based on the evolving reference policy.

### Conceptual Insights

Our approach to optimizing models in this project is guided by several foundational ideas:

- In SFT, we optimize the model to output the exact sequence of tokens, whereas in GRPO (even if the text is identical), we optimize the parameters to produce any sequence that maximizes reward.

- This approach effectively bootstraps the model onto itself, as it uses its own policy as a baseline for improvement.

- The more capable the base model, the more significant the returns from our RL training, provided the RL environment is sufficiently challenging (and not overly so).