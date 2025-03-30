import re

# Reasoning format response reward functions

def count_xml(text:str)->float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count -= len(text.split("<think>\n")[0])*0.001  # penalize slightly for thinking before "think"
        count += 0.25
    if text.count("\n</think>\n") == 1:
        count += 0.25
    if text.count("\n<answer>\n") == 1:
        count += 0.25
    if text.count("\n</answer>") == 1:
        count += 0.25
        count -= (len(text.split("\n</answer>")[-1]))*0.001  # penalize slightly for answering after "answer"
    return max(count, 0.0)

def partial_reasoning_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function to guide models to use a specific format for reasoning. Does not account for ordering and placement of tags."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def strict_reasoning_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for strict adherence to our specific reasoning format."""
    pattern = (
        r"^<think>\n"
        r"([\s\S]+?)\n"
        r"</think>\n"
        r"<answer>\n"
        r"([^\n]+)\n"
        r"</answer>\s*$"
    )
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1.0 if match else 0.0 for match in matches]