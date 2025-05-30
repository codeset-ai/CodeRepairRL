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

def normalize_file_diff(file_diff):
    """Extract filename and changed lines for comparison."""
    import re
    lines = file_diff.splitlines()
    # Extract filename from first line
    match = re.search(r'diff --git a/(.*) b/', lines[0]) if lines else None
    filename = match.group(1) if match else ""
    # Get only the actual changes (+ and - lines)
    changes = [l for l in lines if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
    return (filename, tuple(changes))

def unified_diff_similarity_reward_func(patch, generated_diff, **kwargs) -> list[float]:
    """Unified diff specific reward function that compares file changes regardless of order."""
    assert len(patch) == len(generated_diff), "Patch and generated diff must have the same length"
    
    scores = []
    for p, g in zip(patch, generated_diff):
        # Parse files from both diffs into dictionaries
        patch_files = {normalize_file_diff(f)[0]: normalize_file_diff(f)[1] 
                      for f in split_diff_by_files(p)}
        gen_files = {normalize_file_diff(f)[0]: normalize_file_diff(f)[1] 
                    for f in split_diff_by_files(g)}
        
        if not gen_files:
            scores.append(0.0)
            continue
            
        # Calculate similarity for each file
        file_scores = []
        for filename in set(patch_files) | set(gen_files):
            if filename in patch_files and filename in gen_files:
                # Use SequenceMatcher to compare the changes
                matcher = SequenceMatcher(None, patch_files[filename], gen_files[filename])
                file_scores.append(matcher.ratio())
            else:
                # File exists in only one diff
                file_scores.append(0.0)
        
        scores.append(sum(file_scores) / len(file_scores) if file_scores else 0.0)
    
    return scores