# Relevant information

## https://www.primeintellect.ai/blog/intellect-2 (decentralized, crowd sourced, inference rollout for GPRO training)


# Research Papers

## https://r2e-gym.github.io/

## https://arxiv.org/pdf/2504.02605

## S∗: Test Time Scaling for Code Generation
**Link:** https://arxiv.org/pdf/2502.14382

## All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning
**Link:** https://arxiv.org/pdf/2503.01067

## rStar-Math
**Authors:** Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, Mao Yang  
**Link:** https://arxiv.org/pdf/2501.04519

<details>
<summary><b>Summary</b></summary>

This paper introduces rStar-Math, a framework that enables small language models (SLMs) to achieve state-of-the-art mathematical reasoning capabilities without distillation from larger models. The approach uses Monte Carlo Tree Search (MCTS) at test time, where a math policy SLM performs search guided by an SLM-based process reward model. The authors introduce three key innovations: (1) a code-augmented Chain-of-Thought data synthesis method that generates verified reasoning trajectories through MCTS rollouts; (2) a novel process reward model training method that avoids step-level score annotation; and (3) a self-evolution recipe where both the policy SLM and process preference model (PPM) iteratively improve. Through four rounds of self-evolution with millions of synthesized solutions for 747k math problems, rStar-Math dramatically improves small models' performance. For example, it boosts Qwen2.5-Math-7B from 58.8% to 90.0% and Phi3-mini-3.8B from 41.4% to 86.4% on the MATH benchmark, surpassing OpenAI's o1-preview. On the USA Math Olympiad (AIME), rStar-Math solves an average of 53.3% of problems, ranking among the top 20% of high school math competitors. This work demonstrates that small models can achieve exceptional reasoning capabilities through carefully designed search and training strategies, without requiring massive model scaling.
</details>

## SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
**Authors:** Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, Sida I. Wang  
**Link:** [https://arxiv.org/pdf/2502.18449](https://arxiv.org/pdf/2502.18449)

<details>
<summary><b>Summary</b></summary>

This paper introduces SWE-RL, the first approach to scale reinforcement learning (RL) for improving LLM reasoning in real-world software engineering tasks. Unlike previous work that focused on competitive coding and math problems, SWE-RL leverages open-source software evolution data (code snapshots, changes, and events like issues and PRs) with a lightweight rule-based reward system. The authors trained Llama3-SWE-RL-70B, which achieves a 41.0% solve rate on SWE-bench Verified, making it the best-performing medium-sized (<100B) LLM to date, comparable to proprietary models like GPT-4o. Notably, despite being trained solely on software engineering tasks, the model demonstrated improved general reasoning capabilities across five out-of-domain tasks including function coding, library use, code reasoning, mathematics, and general language understanding.
</details>

## AlphaMaze: Enhancing Large Language Models' Spatial Intelligence via GRPO
**Authors:** Alan Dao (Gia Tuan Dao), Dinh Bach Vu  
**Link:** [https://arxiv.org/pdf/2502.14669](https://arxiv.org/pdf/2502.14669)

<details>
<summary><b>Summary</b></summary>

This paper presents a novel two-stage training framework to equip standard LLMs with visual spatial reasoning abilities for maze navigation. The approach first uses Supervised Fine-Tuning (SFT) on tokenized maze representations to teach step-by-step movement prediction, followed by Group Relative Policy Optimization (GRPO) with a carefully crafted reward function to refine sequential decision-making. While baseline models completely failed at maze navigation (0% accuracy), the SFT-trained model achieved 86% accuracy, and further GRPO fine-tuning boosted performance to 93%. The authors observed that GRPO fostered more robust and self-corrective reasoning, including emergent chain-of-thought behaviors. This work demonstrates how techniques originally developed for language reasoning (like those in DeepSeek-R1) can be successfully adapted to enhance spatial reasoning in LLMs, with potential applications in robotics, autonomous navigation, and other domains requiring integrated visual and sequential reasoning.
</details>

## DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
**Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo  
**Link:** [https://arxiv.org/pdf/2402.03300](https://arxiv.org/pdf/2402.03300)

<details>
<summary><b>Summary</b></summary>

This paper introduces Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm designed to boost the reasoning abilities of language models. GRPO is a variant of PPO that forgoes a separate critic, instead computing a baseline from grouped sample rewards, greatly reducing the resource overhead of RL training. Applied to a math-focused 7B model (DeepSeekMath), GRPO significantly improved performance on mathematical problem benchmarks (e.g. raising GSM8K accuracy from 82.9% to 88.2%). The work also unifies various alignment techniques (RFT, DPO, PPO, GRPO) under a common framework, highlighting their relationships as direct or simplified RL methods. GRPO demonstrates a novel RL approach to enhance reasoning in LLMs. By eliminating the need for a value critic and leveraging group-based rewards, it shows how to efficiently fine-tune models for complex reasoning tasks. This approach is directly relevant to eliciting step-by-step reasoning in LLMs and could be adapted to program repair scenarios, where sparse rewards (e.g. code passes tests or not) make traditional RL challenging. The success of GRPO in improving math reasoning suggests that similar RL-driven fine-tuning can help an LLM learn to reason through code fixes or debugging steps with limited feedback signals.
</details>

## DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**Authors:** Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, et al. (DeepSeek-AI Team)  
**Link:** [https://arxiv.org/pdf/2501.12948](https://arxiv.org/pdf/2501.12948)

<details>
<summary><b>Summary</b></summary>

This work presents a multi-stage RL training pipeline that produces "DeepSeek-R1," a language model with emergent reasoning skills trained entirely through reinforcement learning. First, a 671B base model (DeepSeek-R1-Zero) is trained from scratch with large-scale RL (no supervised fine-tuning), yielding strong reasoning behaviors but issues like mixed languages. Then DeepSeek-R1 is obtained by incorporating a cold-start phase (some initial supervised data) before RL, stabilizing training. The resulting model achieves reasoning performance on par with OpenAI's proprietary model (o1-1217). Notably, the team open-sourced both R1 and R1-Zero, along with six distilled models ranging from 1.5B to 70B parameters derived from R1's training (built on Qwen and Llama backbones). DeepSeek-R1 is a milestone showing that pure RL can foster general reasoning in LLMs without extensive human demonstrations. Its multi-stage approach (RL-only pretraining, then RL with a guided start) and the successful distillation of a huge RL-trained model into smaller models provide a blueprint for building reasoning-focused LLMs. For a project on program repair, this suggests that an RL-trained model could internalize complex debugging strategies, and those skills can be transferred to smaller, more practical model sizes. The open release of DeepSeek-R1 and its distilled versions offers valuable resources and baselines for applying RL to reasoning in tasks like code correction.
</details>

## Code Security Vulnerability Repair Using Reinforcement Learning with Large Language Models
**Authors:** Nafis Tanveer Islam, Mohammad Bahrami Karkevandi, Peyman Najafirad  
**Link:** [https://arxiv.org/pdf/2401.07031](https://arxiv.org/pdf/2401.07031)

<details>
<summary><b>Summary</b></summary>

This work focuses on secure program repair, using RL to train an LLM to fix vulnerabilities in code. Standard fine-tuning often fails to inject small but critical security patches (like null checks or input sanitization) because the model prioritizes reproducing the original functional code (minimizing loss) and neglects minor edits. To address this, the authors propose an RL-based training regime that rewards the model for adding security-improving lines while preserving functionality. They design a combined semantic and syntactic reward: one part encourages correct program behavior (passing tests), and another gives extra credit when the fix includes the required security code patterns. This guides the LLM to produce code fixes that not only solve the problem but also harden security. This paper applies RL to a specific kind of program repair – fixing security bugs – highlighting how carefully crafted reward signals can induce an LLM to follow complex repair requirements. The two-tier reward (functionality + security) demonstrates how to guide LLMs to produce solutions that meet multiple criteria beyond simply minimizing loss.
</details>

## Improving Multi-Step Reasoning Abilities of Large Language Models with Direct Advantage Policy Optimization
**Authors:** Jiacai Liu, Chaojie Wang, Chris Yuhao Liu, Liang Zeng, Rui Yan, Yiwen Sun, Yang Liu, Yahui Zhou  
**Link:** [https://arxiv.org/pdf/2412.18279](https://arxiv.org/pdf/2412.18279)

<details>
<summary><b>Summary</b></summary>

This paper introduces Direct Advantage Policy Optimization (DAPO), an offline RL algorithm tailored for multi-step reasoning in LLMs. DAPO addresses two key challenges in using RL for reasoning: sparse final rewards and unstable training with standard actor-critic methods. Instead of only giving a reward at the end of a solution, DAPO trains a critic to predict the correctness of each intermediate reasoning step, providing dense feedback to the policy at every step. The actor (LLM) and critic are updated separately (avoiding the fragile co-training of PPO). Trained on mathematical proofs and code reasoning queries, DAPO produced models with markedly enhanced step-by-step reasoning accuracy. Experiments show that DAPO improved both math problem solving and code-related tasks over strong baselines, for models that had either been supervised-tuned or already RL-tuned.
</details>

## Offline Reinforcement Learning for LLM Multi-Step Reasoning (OREO)
**Authors:** Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, Yi Wu  
**Link:** [https://arxiv.org/pdf/2412.16145](https://arxiv.org/pdf/2412.16145)

<details>
<summary><b>Summary</b></summary>

The authors propose OREO (Offline Reasoning Optimization), an offline RL method to improve the multi-step reasoning of LLMs without requiring online interactions. They point out limitations of prior alignment methods like Direct Preference Optimization (DPO) for reasoning, such as needing curated preference pairs and providing no mechanism for credit assignment across a long solution. OREO tackles this by jointly training a policy (the LLM) and a value function via a soft Bellman equation, an idea from maximum-entropy RL. This enables the model to learn from reasoning trajectories with sparse rewards by internally propagating value estimates for intermediate steps. In evaluations, an OREO-trained model outperformed other offline methods on complex reasoning benchmarks – from math word problems to an embodied agent task (ALFWorld) – indicating better planning and stepwise deduction. The learned value function can also be used at inference time (via tree search or lookahead) to further boost performance without additional training. OREO exemplifies how offline RL can be leveraged to enhance reasoning, which is useful when interactive environments (like a code executor or user feedback) are limited. In the context of program repair, one could train on logged data of code attempts and outcomes, using OREO's technique to assign credit to each edit or reasoning step that led to a successful fix. Its success on both mathematical reasoning and even non-language planning tasks suggests a general improvement in the model's ability to handle sequential decision-making. This connects to the project by illustrating a way to instill better long-horizon reasoning in an LLM (like debugging through multiple steps) using only existing data, avoiding the need for live reward queries while still reaping the benefits of RL-style optimization.
</details>

## Reasoning Language Models: A Blueprint
**Authors:** Maciej Besta, Julia Barth, Eric Schreiber, Ales Kubicek, Afonso Catarino, Robert Gerstenberger, et al.  
**Link:** [https://arxiv.org/pdf/2501.11223](https://arxiv.org/pdf/2501.11223)

<details>
<summary><b>Summary</b></summary>

This work provides a comprehensive survey and framework for Reasoning Language Models (RLMs) – advanced LLMs augmented with reasoning abilities. It identifies that cutting-edge models like OpenAI's o1 and DeepSeek's models combine multiple components (LLMs, search algorithms, reinforcement learning, etc.) in complex pipelines, which makes them powerful but also hard to reproduce. As a solution, the authors propose a modular blueprint for building RLMs, breaking down the system into distinct parts: reasoning structures (chain-of-thought sequences, tree search, graphs of ideas), reasoning strategies (e.g. beam search, Monte Carlo Tree Search), RL elements (policy/value networks, reward models), and supervision types (outcome-based rewards vs. process supervision). They show how recent methods (like LLaMA-Berry, Journey Learning, Graph-of-Thought, Alibaba's QwQ) fit into this framework, illustrating common patterns. They even introduce a prototypical implementation called x1, to rapidly experiment with different reasoning modules. The blueprint highlights best practices such as multi-phase training (first train a policy model, then a value model) and ensuring the model is familiar with the training distribution of reasoning steps.

For someone researching reasoning in LLMs (like through program repair), this blueprint is a high-level guide that places reinforcement learning in context with other techniques. It emphasizes that RL is one piece of a larger puzzle: effective reasoning may also require search procedures (e.g. exploring multiple candidate fixes), structured thought (like maintaining a chain-of-thought about code execution), and possibly separate value estimation (to judge partial solutions). By drawing analogies to methods across domains, it can inform the project how to integrate RL with techniques like search or knowledge retrieval to build a more effective code reasoning agent. In essence, this paper acts as a map of the design space, helping to ensure the approach to eliciting reasoning (via RL or otherwise) is informed by a broad view of current research.
</details>

## NExT: Teaching Large Language Models to Reason about Code Execution
**Authors:** Ansong Ni, Miltiadis Allamanis, Arman Cohan, Yinlin Deng, Kensen Shi, Charles Sutton, Pengcheng Yin  
**Link:** [https://arxiv.org/pdf/2404.14662](https://arxiv.org/pdf/2404.14662)

<details>
<summary><b>Summary</b></summary>

NExT is an approach by DeepMind to enhance an LLM's reasoning on programming tasks by integrating execution traces into its thought process. Rather than relying solely on static code, NExT provides the model with information from running the code (e.g. values of variables at runtime, error messages) and trains it to incorporate this into chain-of-thought explanations. The method uses self-training: the model generates its own reasoning steps and observes execution results, then learns from those augmented rationales without requiring manual annotations. By iteratively refining its reasoning with real execution feedback, an LLM (based on PaLM 2) dramatically improved at debugging and fixing code. On two program repair benchmarks (Google's MBPP and OpenAI's HumanEval bug-fix tasks), NExT boosted the code fix rate by 26.1% and 14.3% absolute, respectively, compared to the baseline model. Importantly, the model's explanations of code behavior became more aligned with actual program logic, as verified by human evaluators.

Relevance: While NExT is not a pure RL method, it tackles the same goal – eliciting better reasoning in LLMs – through a clever analog: using execution feedback as a training signal. This is highly relevant to program repair, since debugging usually involves running code to see what went wrong. The idea of naturalizing execution traces into the LLM's reasoning can complement RL approaches: for example, an RL agent fixing code could use execution results as part of its reward or state representation. NExT shows that giving an LLM the ability to "think like a debugger" (by seeing runtime information) yields substantial improvements in fixing errors. For the project, this suggests incorporating tools (like code execution or tests) into the training loop – either via explicit rewards or self-training – to encourage the model to reason through the semantics of code, not just its syntax. It's an analogy to RL in that the model is learning from interactive feedback (execution outcomes) to improve its policy of writing correct code.
</details>

## DeepSeek-R1 Distilled Models (Qwen2.5 Series) – System Card
**Authors:** DeepSeek-AI Team & Alibaba Qwen Team  
**Link:** [DeepSeek-R1 Distillation Release](https://github.com/deepseek-ai/DeepSeek-R1)

<details>
<summary><b>Summary</b></summary>

Alongside the DeepSeek-R1 paper, the authors released a suite of open-source models that pack DeepSeek's reasoning prowess into smaller architectures. These include models based on Qwen2.5, an Alibaba 14–32B LLM series tuned for strong knowledge, coding, and math skills. Compared to earlier versions, Qwen-2.5 offers notable boosts in code understanding and long-context handling (up to 128K tokens), and much improved instruction following and structured output generation. Using Qwen2.5-32B as a base, DeepSeek's team distilled the large 671B DeepSeek-R1 into a 32B model that achieves state-of-the-art results among models its size. For instance, DeepSeek-R1-Distill-Qwen-32B attains a Codeforces coding competition rating of 1691 (the best of any distilled model, rivaling OpenAI's tuned 35B model) and excels on reasoning benchmarks like AIME math (83.3% correct) and LiveCodeBench programming tasks. Similar distilled models were released at 1.5B, 7B, 14B, and even a distilled 70B Llama, all trained on the reasoning data generated by the DeepSeek-R1 process. These system cards detail that the distilled models maintain strong reasoning capabilities thanks to the transfer of reasoning patterns from the large model. They also note any changes (e.g. modified tokenizers or configs) and recommend using the provided settings for best performance. These system and model cards are valuable references as they illustrate how a high-performing reasoning model can be compressed into smaller ones without losing too much capability. For the project, examining Qwen2.5 and DeepSeek's distilled models provides insight into the backbone model qualities that favor reasoning (Qwen2.5's coding and math-oriented pretraining) and the effectiveness of distillation in retaining reasoning chains. In practice, this means one could leverage these released checkpoints or mimic their distillation approach to build a program repair model: start with a capable base (like Qwen2.5-Math for mathematical reasoning or code understanding) and fine-tune it with an RL or feedback signal, possibly distilling from a larger model if available. The system cards also discuss the limits and intended uses of each model, which helps to understand how far one can push them in tasks like code repair and what adjustments might be needed (e.g. shorter context, certain prompt formats). In summary, DeepSeek's model cards for Qwen2.5-based distillations connect the research to practical, use-case-ready models that can be directly evaluated or adapted in the domain of automated code reasoning and repair.
</details>