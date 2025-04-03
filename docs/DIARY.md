# Project Diary and Planning

### Ideas

#### 1. Enhance trl/GRPOTrainer with Agentic Capabilities
- Integrate `aider.Coder.chat` to replace the standard `model.generate` functionality
- Leverage aider's built-in agentic coding tools for:
  - Repository exploration and understanding
  - Multi-file change management
- Outcome-supervised reward modeling, reward the final patch generated, not the process of creating it
- Essentially just running multiple coding agents in parallel and reinforce behaviors that lead to superior outcomes

#### 2. Test-Driven Reinforcement Learning
- Implement a reward system based on passing test cases rather than similarity to reference solutions
- Deploy scalable Kubernetes infrastructure to parallelize test case execution across multiple model outputs
- Benefits:
  - Rewards functional correctness while allowing creative problem-solving approaches
  - Provides greater flexibility compared to "golden patch" approaches
  - Encourages models to develop diverse solution strategies that still satisfy requirements

#### The stack.py, LLM generated versions of the Oracle patch

### Backlog

<details>
<summary>General Research Tasks</summary>

- [ ] Look at how SWE-Bench works exactly, should we train for that behavior end2end?
- [ ] Read S∗: Test Time Scaling for Code Generation: https://arxiv.org/pdf/2502.14382
- [ ] Read MUFIN **paper**
- [ ] Update DATASETS.md with everything
  - From overleaf
  - From emails
  - From docs (both mine and minutes)
</details>

<details>
<summary>Advanced SWE Tasks (Conditional)</summary>
- [ ] Replicate the "agentic workflow" of SWE-agent and similar 
  - Find out how they normally put the codebase into the context window
  - How they do function calling
  - Otherwise we could just run repomix
- [ ] Multi file repair
</details>


## April 1 - April 7, 2025

#### Tasks:
- [x] Jobscripts vastly simplified
- [x] Make hf cache (and uv if simple) point to my project directory instead of my home directory
- [ ] Double check recommended flast-attn version for vllm
  - Now I get "backend not found" errors
- [ ] Train Qwen-2.5-Coder-32B on stack "repair" on SLURM
  - Main concern is whether the two apptainers can talk to each other (the vLLM server opens a local unicorn server, I'd assume that it does)
  - Could solve this on the python side, simplify the .sh side and use one script which detects gpu resources and launches training/inference nodes accordingly
- [ ] Check on verifiers, is it better now?
  - vLLM support?
- [ ] Replace vllm_client.generate with paralellized Aider instances
  - Main concern is just whether trl-serve launches an OpenAI compatible server or something more minimal.
- [ ] Create SWE-Bench "repair" dataset
  - Maybe best to just call the fast clone method I made at runtime
- [ ] Setup Defect4J as reward signal and train on it
- [ ]

## March 24 - March 30, 2025

#### Tasks
- [x] Remove unsloth for now, that library is a nightmare to work with
  - Simplifies the config by a lot
- [x] Reward callbacks only receive prompt and completion as args, rest is kwargs
- [x] Simplify diff
  - Just search/replace, refactor for possible future multi-file edits
- [x] Simplify rewards  
  - All scaled between 0-1, reward weighting done in the script
- [x] Forked trl
  - Ideas such as integrating Aider will require it
  - For now, the only change is better wandb logging
- [x] Support newest trl version
  - Something changed in the last few weeks in trl
  - Need to run vllm serve myself, makes sense
  - Simply need to add trl vllm-serve to the .sh scripts
  - Post:
    - We kind of need multi gpu for this to work well, I get cryptic NCCL errors when trying to force it to occupy only 1 gpu
    - For simple testing we can simply do uv run -m src.train_grpo grpo.use_vllm=false
    - But for larger scale training we run the .sh scripts
- [x] Deepspeed / accelerate support
  - Post:
    - Needs testing, should be pretty good
    - Opted for "ZeRO-Stage 3" deepspeed, where optimizer state/gradients/model parameters are all sharded between GPUs
    - This isn't really necessary for training LoRAs, but I'd like to move past those ultimately  
- [x] Stack "repair" on KTH DGX-100
- [x] Added .from_unified_diff to SearchReplace
- [x] Added git utils for datasets which require us to clone repositories
  - Tested a few different methods, the fastest was to create an empty repo, add the origin, then fetch only the relevant commit 

#### Ideas
We could simply:
1. host vLLM
2. run Aider Coder or similar on a repo, tasked with e.g. fixing a bug
3. it returns a diff which we score, and we weigh the entire conversation history with the rewards
I.e. directly reinforce performance when using aider. Requires slight refactoring of trl.GRPOTrainer, but not too bad.
This vastly simplifies performing multi file edits and stuff. Perhaps issues with batching



## March 17 - March 23, 2025
Was a bit sick during the weekened which is when I am normally the most productive

#### Tasks
- [x] Fix apptainer and test_job.sh
  - I know it works to cp ttc.sif to my project directory then do apptainer exec ttc.sif uv run ...
  - Maybe some discrepancy between that and what test_job.sh is doing
  - Or some complexity I'm not aware about in srun
- [x] Fix some refactor mismatches
- [x] Stack implementation dataset
  - It's not exactly the type of dataset we are looking for but its pretty good
- [x] Read LLM Post-Training: A Deep Dive into Reasoning Large Language Models: https://arxiv.org/pdf/2502.21321

#### Ideas
- What if we use AST to extract docstrings of python code on the Stack.
  - Sufficiently detailed docstrings can be mutated (via LLM) into a detailed description of the desired functionality of a function or class
  - We then remove said function/class but provide the rest of the code as context


## March 10 - March 16, 2025

#### Supervisor Updates
- Using test cases as rewards may require alot of SWE to run each test case on Kubernetes
  - Issues with scalability

#### Tasks
- [x] Fix diff.py, it is VERY sloppy
- [x] Fix reward.py, it sloppy
  - Get rid of get_diff entirely
  - Make correctness and quality rewards as separate as possible
  - Merge strict quality with partial quality, since the latter smoothly transitions from 0-1
  - Also add a method to diff.py to compare two diffs of the same type 
- [x] Side quest: Fast QwQ-32B inference on 1 of KTH's H100
- [x] Check out verifiers, seems like exactly what I need, could even contribute to the repo
  - paralell execution env where we can call dockerized tests?
- [x] Check out lighteval, would be cool to check e.g. humaneval before and after
  - looks good, much better than deepeval which I was using before


#### Thoughts
- Should we do have an agent scaffolding like SWE-Agent / OpenHands / Agentless
- Would be really cool to reinforce tool usage but that requires a lot of engineering
- Bug fix diff matching is probably almost equivalent to running the tests
  - At least for simple bugs which only require changing a couple lines
- Perhaps a more interesting task is implementing something new with well defined behaviour
  - Then oracle patch diff matching is a vastly different objective from passing the tests
  - Can we create a synthetic dataset for this?


## March 3 - March 9, 2025

#### Supervisor Updates
- Notes from meeting:
  - I need to scope out my project more, decide what task I am optimizing for exactly
  - Is that code execution RL? 
  - Will I be doing a ML paper? Data curation and cleaning paper / agentic coder paper?
  - Am I targeting SWE-Bench? RepairBench?

#### Tasks
- [x] Scope out my project
- [ ] Look at how SWE-Bench works exactly, should we train for that behavior end2end?
- [ ] Fix SLURM scripts (get a basic training run going)

#### Thoughts
- Should I keep on using GRPOTrainer?
- Should I fork verifiable envs or create my own (execution consideration)
