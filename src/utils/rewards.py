import re

from src.utils.diff import SearchReplaceDiff


def extract_xml_answer(text:str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return "N/A"


# FYI: rewards are calculated as callbacks which receive the keywords prompts, completions and all the columns of your dataset as kwargs

##################################################################################################################  
# Reasoning style response reward functions
##################################################################################################################

def count_xml(text:str)->float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count -= len(text.split("<think>\n")[0])*0.001  # penalize slightly for thinking before "think"
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
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
    return [0.5 if match else 0.0 for match in matches]

##################################################################################################################
# Detection specific reward functions
##################################################################################################################

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the extracter answer matches the ground truth answer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    return [1.0 if ext == a else 0.0 for ext, a in zip(extracted_responses, answer)]

##################################################################################################################
# Repair specific reward functions
##################################################################################################################

def diff_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks the quality of the extracted diff format between 0.0 and 1.0."""
    
    contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    diffs_batch = [SearchReplaceDiff.extract_all(diff_str) for diff_str in contents]  # batch of a list of diffs (for potentially multiple files)
    
    return [
        sum(diff.validate_quality() for diff in diffs)/(len(diffs) or 1)
        for diffs in diffs_batch
    ] 

def diff_similarity_reward_func(prompts, completions, diffs, **kwargs) -> list[float]:
    """Reward function that checks if the sequence of search/replace diffs matches the reference diffs."""
    
    contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    generated_diffs_batch = [SearchReplaceDiff.extract_all(diff_str) for diff_str in contents]
    reference_diffs_batch = [SearchReplaceDiff.extract_all(diff_str) for diff_str in diffs]  # TODO: support having multiple reference diffs
    
    # TODO: ugly, support multi file diffs better
    rewards = []
    for ref_diffs, gen_diffs in zip(reference_diffs_batch, generated_diffs_batch):
        assert len(ref_diffs) == len(gen_diffs), "Number of reference diffs and generated diffs must be the same"
        per_file_similarity = []
        for ref_diff, gen_diff in zip(ref_diffs, gen_diffs):
            per_file_similarity.append(ref_diff.similarity(gen_diff))
        rewards.append(sum(per_file_similarity)/(len(per_file_similarity) or 1))

    return rewards