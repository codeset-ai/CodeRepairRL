# We can potentially use the DeepSeek-R1 reasoning traces on RepairBench to bootstrap our own reasoning traces.
# We can achieve this by first performing SFT on a model of our choice on that data

# 1. get the patches and whether they are correct from the repairbench-cache git submodule
# 2. perform SFT on a model of our choice on that data
# 3. either GRPO by sampling from the new model or by lying to it that the reasoning traces from R1 are its own reasoning traces

# Maybe there is a large published dataset of reasoning traces, I know the open-r1 team are curating them atleast.