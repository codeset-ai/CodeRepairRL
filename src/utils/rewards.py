import re
import difflib
from src.utils.diff import is_valid_diff_format


def extract_xml_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return "N/A"

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion is correct.
    Gives some points if the model outputs the correct CWE, but with extra text.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [
        2.0 if ext == a else 0.5 if ext.split(":")[0] == a else 0.0
        for ext, a in zip(extracted_responses, answer)
    ]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n([\s\S]+?)\n</think>\n<answer>\n([^\n]+)\n</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
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

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


### Repair specific reward functions


def count_diff_format(text) -> float:
    """
    Calculate a partial reward for diff format based on presence of markers.
    
    Args:
        text: The text to analyze for diff format markers
        
    Returns:
        A score between 0.0 and 0.3 indicating how well the text follows diff format
    """    
    score = 0.0
    
    if "<<<<<<< SEARCH" in text:
        score += 0.1
    
    # Check for divider
    if "=======" in text:
        score += 0.1
    
    # Check for replace markers
    if ">>>>>>> REPLACE" in text:
        score += 0.1
    
    return score

def partial_diff_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that gives partial credit for diff format.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of scores between 0.0 and 1.0 for each completion
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_diff_format(c) for c in contents]

def strict_diff_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that gives full credit for diff format.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [1.0 if is_valid_diff_format(c) else 0.0 for c in contents]


def diff_similarity_reward(reference_diff: str, generated_diff: str) -> float:
    """
    Calculate similarity between two git diffs using SequenceMatcher.
    Also checks if the generated diff is in valid format.
    
    Args:
        reference_diff: The ground truth git diff
        generated_diff: The generated git diff
    
    Returns:
        A similarity score between -1.0 and 1.0, where:
          1.0 means identical diffs
          Values between 0.0-1.0 represent partial similarity
          -1.0 means invalid diff format
    """
    # First check if the generated diff is in valid format
    if not is_valid_diff_format(generated_diff):
        return -1.0
    
    # Split both diffs into lines for better comparison
    ref_lines = reference_diff.splitlines()
    gen_lines = generated_diff.splitlines()
    
    # Use SequenceMatcher to calculate similarity ratio
    seq_matcher = difflib.SequenceMatcher(None, ref_lines, gen_lines)
    similarity = seq_matcher.ratio()
    
    return similarity

def diff_similarity_reward_func(completions, reference_diffs, **kwargs) -> list[float]:
    """
    Reward function that compares generated diffs with reference diffs using sequence matching.
    
    Args:
        completions: List of model completions
        reference_diffs: List of reference git diffs to compare against
        
    Returns:
        List of similarity scores between -1 and 1 for each completion:
          - Scores between 0-1 indicate valid diffs with varying similarity
          - Score of -1 indicates invalid diff format
    """
    # Extract content from completions
    contents = [completion[0]["content"] for completion in completions]
    
    # Extract diffs from completions if needed
    # This assumes the diff is the full content, but you might need to extract it
    # using a regex pattern if the completion contains other text
    
    # Calculate similarity for each completion-reference pair
    similarities = [
        diff_similarity_reward(ref_diff, gen_diff) 
        for ref_diff, gen_diff in zip(reference_diffs, contents)
    ]
    
    return similarities