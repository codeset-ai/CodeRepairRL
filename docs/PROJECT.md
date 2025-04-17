# Project Details for CodeRepairRL

## Implementation Note

This project uses a [custom fork of TRL](https://github.com/BjarniHaukur/trl) (Transformer Reinforcement Learning), originally enhanced for better Weights & Biases logging. The fork has evolved to support integrating coding agents directly into the reinforcement learning loop, with contributions being prepared for potential integration into the main TRL repository.

## Technical Innovation: Agent-in-the-Loop Reinforcement Learning

Our project pioneers a novel approach that integrates agent frameworks directly into the reinforcement learning training loop. By implementing an OpenAI-compatible API server with vLLM that supports asynchronous token streaming, we enable:

1. **Multi-step, interactive agent training** - Rather than single-pass generation, agents can perform complex sequences of actions including repository exploration, code editing, and command execution.

2. **Parallelized inference with independent control** - Multiple agents operate concurrently on the same tasks, with independent control over their generation process (e.g., one might execute bash commands while another writes code).

3. **Direct reinforcement in realistic environments** - Training occurs in the same environment where models will be deployed, rewarding effective tool use and complex reasoning paths that lead to correct solutions.

This approach significantly lowers the barrier to entry for research on tool-using, environment-interactive agent training, bringing open-source capabilities closer to what we believe large AI labs are implementing internally.

## Objectives

### Code Repair

Our primary objective remains generating accurate code patches to fix specific software issues. Initially, we reward the model based on the accuracy of the generated diffs using a search/replace approach that compares model-generated diffs against oracle solutions. With our agent-in-the-loop approach, we can now explore more complex multi-file edits while maintaining evaluation simplicity. Ultimately, we aim to use software tests as the oracle for correctness, though this presents challenges due to the extensive compute requirements for running comprehensive test suites.

### Code Implementation from Specification

We also explore the task of implementing code from specifications using our stack.py dataset. This represents a related but distinct challenge from code repair, with fewer existing datasets in this category. The model must generate a complete implementation based on specifications, testing its understanding of requirements, adherence to contextual style, and ability to produce functional code. Our agent integration approach is particularly valuable here, as it allows for incremental implementation and self-testing.

### Vulnerability Classification

As a secondary objective, we address classification of software vulnerabilities. This task evaluates our model's ability to reason about complex software vulnerabilities, effectively creating a test-time compute-enabled, process-verifiable classifier. Unlike traditional black-box models, our approach provides explicit reasoning paths, enhancing transparency and trustworthiness.

### Benchmark Performance Evaluation

A key goal of this project is to observe and measure performance improvements on established benchmarks including SWE-Bench, Aider-polyglot, and similar evaluation frameworks. These benchmarks provide standardized measurement of code understanding, editing, and implementation capabilities, allowing us to quantify the effectiveness of our agent-in-the-loop reinforcement learning approach compared to existing methods.

For detailed run results and progress on our objectives, see our [WandB project page](https://wandb.ai/assert-kth/TTC).

## Post-Training Perspective

Post-training of LLMs is a complex, multivariate process consisting of many specialized steps across various domains, ultimately resulting in powerful general-purpose models. While achieving a comprehensive post-training pipeline is beyond the scope of a master's thesis, our project aims to demonstrate a significant performance improvement in the specific domain of code repair through agent-in-the-loop training. We believe such targeted improvements represent one valuable step in the broader post-training stage of frontier models.

## Experimental Plan & Benchmark Rationale

Our agent‑in‑the‑loop code‑repair thesis hinges on **showing real generalisation, not memorisation**.  
To that end we split data into **(i) a medium‑size, fully‑reproducible training set** that lets the agent learn tooling skills, and **(ii) two tougher, newer evaluation suites** that expose it to unseen projects and build systems.

| Phase | Corpus | Language(s) | Why it’s in‑scope |
|-------|--------|-------------|-------------------|
| **Train / fine‑tune** | **SWE‑Gym** (≈ 2.4 k tasks) | Python | RL‑ready, containerised tasks—perfect for teaching interactive tool use. |
| | **Defects4J v2.0** (835 bugs) | Java | Decade‑long APR baseline; ensures direct comparability with the literature. |
| **Evaluate (primary)** | **SWE‑Bench‑Verified** (≈ 2.3 k bugs) | Python | Harder, multi‑file patches in modern repos; measures generalisation beyond SWE‑Gym. |
| | **GitBug‑Java** (199 recent bugs) | Java | 2020‑2023 GitHub issues with Gradle/Maven builds; covers modern Java ecosystems that Defects4J lacks. |
| **Evaluate (sanity / optional)** | **BugsInPy** + **QuixBugs** | Python & Java | Quick cross‑language smoke tests; catch trivial regressions before full runs. |

**Why this split works**

* **Language breadth** – Python + Java covers the two most‑studied code‑repair ecosystems and showcases cross‑language transfer.  
* **Temporal gap** – Training data stops at 2018 (Defects4J) and curated snapshots (SWE‑Gym); evaluation bugs are 2020 – 2024, reducing leakage risk.  
* **Task diversity** – From single‑hunk unit‑test fixes (Defects4J) to multi‑file repo repairs (SWE‑Bench, GitBug‑Java), giving a holistic picture of the agent’s capabilities.  
* **Comparability** – Nearly every APR paper reports Defects4J; SWE‑Bench is the emerging LLM standard. Using both lets reviewers benchmark us instantly.

Success criterion: **≥ 5 pp absolute improvement** over strong zero‑shot LLM baselines on SWE‑Bench‑Verified *and* any noticeable lift on GitBug‑Java (where no fine‑tuned agent numbers exist yet). Achieving that validates the thesis claim that agentic RL on realistic tasks yields transferable repair skills.


## Compute Efficiency

Since LLMs are inherently large, training them can be challenging for individual researchers. Our project has moved to a streamlined approach using DeepSpeed and Accelerate for distributed training. We use "ZeRO-Stage 3" DeepSpeed configuration, where optimizer state, gradients, and model parameters are all sharded between GPUs.

## Definitions and Intuitions

In our project, we use specific terms that are crucial for understanding our approach:

- **KL-divergence (Kullback-Leibler divergence)**: A fundamental metric in information theory that measures how one probability distribution differs from another. In our reinforcement learning setup, we use it to quantify how much our updated policy ($\pi_{t+1}$) diverges from our reference policy ($\pi_t$) as the training progresses. This helps ensure our model learns gradually and stably.

- **Loss**: In the context of our model, 'loss' refers to the discrepancy between the reference policy and the update policy. It is a measure of how much the model's predictions deviate from the expected outcomes based on the evolving reference policy.

### Conceptual Insights

Our approach to optimizing models in this project is guided by several foundational ideas:

- In SFT, we optimize the model to output the exact sequence of tokens, whereas in GRPO (even if the text is identical), we optimize the parameters to produce any sequence that maximizes reward.

- This approach effectively bootstraps the model onto itself, as it uses its own policy as a baseline for improvement.

- The more capable the base model, the more significant the returns from our RL training, provided the RL environment is sufficiently challenging (and not overly so).

- By training agents in realistic environments with real tool use, we reinforce not just token generation but effective problem-solving strategies and environmental interactions. 