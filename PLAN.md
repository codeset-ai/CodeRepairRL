# Project Planning



## March 10 - March 16, 2025

### Checkpoint: Monday, March 10, 2025

#### Supervisor Updates
- Using test cases as rewards may require alot of SWE to run each test case on Kubernetes
  - Issues with scalability

#### Tasks
- [ ] Fix SLURM scripts (get a basic training run going)
- [ ] Train on SWE-Bench-lite on diff matching
- [ ] Setup Defect4J as reward signal and train on it

#### Thoughts
- Should we do have an agent scaffolding like SWE-Agent / OpenHands / Agentless
- Would be really cool to reinforce tool usage but that requires a lot of engineering



## March 3 - March 9, 2025

### Checkpoint: Monday, March 3, 2025

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
