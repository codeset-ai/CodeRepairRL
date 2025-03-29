# All reward functions should return a list of values ranging from 0.0 to 1.0, reward weighting is handled in the trainer

import re

from src.utils.diff import SearchReplaceDiff


def extract_xml_answer(text:str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return "N/A"

# NOTE: Currently we only support single file diffs
def extract_markdown_block(response: str) -> str:
    """Extract the first code block from a markdown response, or return the response itself if no code block is found."""
    match = re.search(r"```(?:.*?)\n(.*?)```", response, re.DOTALL)
    return match.group(1) if match else response

# FYI: rewards are calculated as callbacks which receive the keywords prompts, completions and all the columns of your dataset as kwargs

##################################################################################################################  
# Reasoning style response reward functions
##################################################################################################################

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

##################################################################################################################
# Detection specific reward functions
##################################################################################################################

def correctness_reward_func(completions, answers, **kwargs) -> list[float]:
    """Reward function that checks if the extracter answer matches the ground truth answer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    return [1.0 if ext == a else 0.0 for ext, a in zip(extracted_responses, answers)]

##################################################################################################################
# Repair specific reward functions
##################################################################################################################

def diff_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks the quality of the extracted diff format between 0.0 and 1.0."""
    
    answer_contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    markdown_contents = [extract_markdown_block(answer_content) for answer_content in answer_contents]
    diffs = [SearchReplaceDiff.from_string(diff_str) for diff_str in markdown_contents]
    
    return [diff.validate_quality() for diff in diffs] 

def diff_similarity_reward_func(completions, diffs, **kwargs) -> list[float]:
    """Reward function that checks if the sequence of search/replace diffs matches the reference diffs."""
    
    answer_contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    markdown_contents = [extract_markdown_block(answer_content) for answer_content in answer_contents]
    generated_diffs_batch = [SearchReplaceDiff.from_string(diff_str) for diff_str in markdown_contents]
    reference_diffs_batch = [SearchReplaceDiff.from_string(diff_str) for diff_str in diffs]
    
    return [diff.similarity(ref_diff) for diff, ref_diff in zip(generated_diffs_batch, reference_diffs_batch)]