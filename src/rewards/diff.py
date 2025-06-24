import re
from difflib import SequenceMatcher

from src.utils.diff import SearchReplaceDiff
from src.utils.extract import extract_xml_answer, extract_markdown_block

# SearchReplaceDiff specific reward functions

def sr_diff_format_reward_func(completions, **kwargs) -> list[float]:
    """SearchReplaceDiff specific reward function that checks the quality of the extracted diff format between 0.0 and 1.0."""
    
    answer_contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    markdown_contents = [extract_markdown_block(answer_content) for answer_content in answer_contents]
    
    # Use the static method directly on the markdown_contents to evaluate format quality
    return [SearchReplaceDiff.validate_quality(diff_str) for diff_str in markdown_contents]

def sr_diff_similarity_reward_func(completions, diffs, **kwargs) -> list[float]:
    """SearchReplaceDiff specific reward function that checks if the sequence of search/replace diffs matches the reference diffs."""
    
    answer_contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    markdown_contents = [extract_markdown_block(answer_content) for answer_content in answer_contents]
    generated_diffs_batch = [SearchReplaceDiff.from_string(diff_str) for diff_str in markdown_contents]
    reference_diffs_batch = [SearchReplaceDiff.from_string(diff_str) for diff_str in diffs]
    
    return [diff.similarity(ref_diff) for diff, ref_diff in zip(generated_diffs_batch, reference_diffs_batch)]

# Unified diff specific reward functions

def split_diff_by_files(diff_text):
    """Split unified diff into individual file diffs."""
    # Split on "diff --git" but keep the delimiter
    parts = re.split(r'(?=^diff --git)', diff_text, flags=re.MULTILINE)
    return [part.strip() for part in parts if part.strip()]

def extract_filename_from_diff(file_diff):
    """Extract filename from a file diff."""
    lines = file_diff.splitlines()
    match = re.search(r'diff --git a/(.*) b/', lines[0]) if lines else None
    return match.group(1) if match else ""

def unified_diff_similarity_reward_func(patch, generated_diff, **kwargs) -> list[float]:
    """Unified diff specific reward function that compares file changes regardless of order."""
    assert len(patch) == len(generated_diff), "Patch and generated diff must have the same length"
    
    scores = []
    for p, g in zip(patch, generated_diff):
        # Split into individual file diffs
        oracle_file_diffs = split_diff_by_files(p)
        gen_file_diffs = split_diff_by_files(g)
        
        # Create filename to diff mapping
        oracle_files = {extract_filename_from_diff(diff): diff for diff in oracle_file_diffs}
        gen_files = {extract_filename_from_diff(diff): diff for diff in gen_file_diffs}
        
        # Calculate similarity for each oracle file
        file_scores = []
        for filename in oracle_files:
            if filename in gen_files:
                # Use SequenceMatcher to compare the raw file diffs
                matcher = SequenceMatcher(None, oracle_files[filename], gen_files[filename])
                file_scores.append(matcher.ratio())
            else:
                # Oracle file not attempted in generated diff
                file_scores.append(0.0)
        
        # Normalize by number of oracle files to prevent reward hacking
        scores.append(sum(file_scores) / len(oracle_files))
    
    return scores

def unified_diff_similarity_reward_func_test(test_patch, generated_diff, **kwargs) -> list[float]:
    return unified_diff_similarity_reward_func(patch=test_patch, generated_diff=generated_diff)

def unified_diff_file_match_reward_func(patch, generated_diff, **kwargs) -> list[float]:
    """Reward function that returns the fraction of patch files correctly identified in generated diff."""
    assert len(patch) == len(generated_diff), "Patch and generated diff must have the same length"
    
    scores = []
    for p, g in zip(patch, generated_diff):
        # Split into individual file diffs and extract filenames
        oracle_file_diffs = split_diff_by_files(p)
        gen_file_diffs = split_diff_by_files(g)
        
        oracle_filenames = {extract_filename_from_diff(diff) for diff in oracle_file_diffs}
        gen_filenames = {extract_filename_from_diff(diff) for diff in gen_file_diffs}
        
        if not oracle_filenames:
            scores.append(1.0 if not gen_filenames else 0.0)
        else:
            scores.append(len(oracle_filenames & gen_filenames) / len(oracle_filenames))
    
    return scores