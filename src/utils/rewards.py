import re
import difflib
from typing import List, Tuple, Optional


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

def is_valid_diff_format(diff_text: str) -> bool:
    """
    Check if a string is in valid git diff format.
    
    Args:
        diff_text: The text to check
        
    Returns:
        True if the text appears to be a valid git diff, False otherwise
    """
    # Basic check for common diff patterns
    # 1. Check for diff header lines
    diff_header_pattern = r'diff --git a/.+ b/.+'
    has_header = bool(re.search(diff_header_pattern, diff_text))
    
    # 2. Check for index lines
    index_pattern = r'index [0-9a-f]+\.\.[0-9a-f]+'
    has_index = bool(re.search(index_pattern, diff_text))
    
    # 3. Check for hunk headers
    hunk_pattern = r'@@ -\d+,\d+ \+\d+,\d+ @@'
    has_hunk = bool(re.search(hunk_pattern, diff_text))
    
    # 4. Check for addition/deletion lines
    changes_pattern = r'[+-][^+-]'
    has_changes = bool(re.search(changes_pattern, diff_text))
    
    # Count how many of these patterns we found
    pattern_count = sum([has_header, has_index, has_hunk, has_changes])
    
    # Consider valid if at least 2 of these patterns are found
    # This is a simple heuristic and can be adjusted
    return pattern_count >= 2

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