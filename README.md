# TTC - Test Time Compute for Program Repair

## Overview

TTC leverages recent advancements in applying Reinforcement Learning (RL) to Large Language Models (LLMs) to train them in domain-specific reasoning. Our ultimate goal is to develop models similar to [RepairLLama](https://arxiv.org/pdf/2312.15698) and [Llama-3-SWE-RL](https://arxiv.org/pdf/2502.18449), which "punch above their weight-class" in terms of parameter count, demonstrating exceptional performance in software engineering benchmarks. This aligns with recent advancements described in the [DeepSeek-R1 paper](https://arxiv.org/pdf/2501.12948).

## Objectives

### Vulnerability Classification

We address a toy problem focused on classifying software vulnerabilities. This task evaluates our model's ability to reason about complex software vulnerabilities, effectively creating a test-time compute-enabled, process-verifiable classifier. Unlike traditional black-box models, our approach provides explicit reasoning paths, enhancing transparency and trustworthiness.

### Code Patch Generation

Our more practical challenge involves generating accurate code patches to fix specific software issues. Initially, we reward the model based on the accuracy of the generated diffs (SequenceMatched diff, ranging from 0 to 1). Ultimately, we aim to use software tests as the oracle for correctness, although this remains challenging due to the extensive time required to run comprehensive test suites.

For detailed run results and progress on both objectives, see our [WandB project page](https://wandb.ai/assert-kth/TTC).

## Key Dataset: PrimeVul

PrimeVul is a very convenient dataset for our project, ideal for testing the [SWE-RL approach](https://arxiv.org/pdf/2502.18449). It includes:

- Paired vulnerable and fixed code snippets.
- Explicit vulnerability descriptions using CWE identifiers.
- Minimal semantic differences between vulnerable and fixed code, isolating the vulnerability.
- Self-contained examples, eliminating the need for external context.

## Compute Efficiency

Since LLMs are inherently large, training them can be challenging for individual researchers. To address this, we optionally include parameter-efficient LoRA training and device optimizations via Unsloth. These optimizations include training at lower precision and using optimized attention implementations, significantly enhancing compute efficiency.

---

## Implementation Details

### Project Structure

- `src/`: Source code for the project
- `scripts/`: Scripts for building, training, and testing
  - `container.def`: Apptainer container definition
  - `build_container.sh`: Script to build the container
  - `train_job.sh`: SLURM script for training
  - `test_job.sh`: SLURM script for testing

### Getting Started

#### Building the Container

To build the Apptainer container:

```bash
./scripts/build_container.sh
```

This creates a `ttc.sif` file in the project root.

#### Running Training Jobs

Submit a training job to SLURM:

```bash
sbatch scripts/train_job.sh --config your_config
```

Options:
- `--config CONFIG`: Specify the configuration to use
- `--unsloth`: Use the unsloth training script
- `--gpus N`: Specify the number of GPUs to use

#### Running Test Jobs

Submit a test job to SLURM:

```bash
sbatch scripts/test_job.sh
```

### Container Details

The container uses uv for dependency management and includes all necessary Python packages specified in the `pyproject.toml` file. Built on Python 3.11, it supports CUDA for GPU training.

### GPU Precision Auto-Detection

The training configuration automatically detects GPU architecture and sets precision:

- **Ampere (SM 8.0) and newer GPUs**: BF16 precision
- **Pascal (SM 6.0) to Turing (SM 7.5) GPUs**: FP16 precision
