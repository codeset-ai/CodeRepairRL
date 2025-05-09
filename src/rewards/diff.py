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

def unified_diff_similarity_reward_func(patch, generated_diff, **kwargs) -> list[float]:
    """Unified diff specific reward function that checks if the sequence of search/replace diffs matches the reference diffs."""

    assert len(patch) == len(generated_diff), "Patch and generated diff must have the same length"

    patch_lines = [p.splitlines() for p in patch]
    generated_lines = [g.splitlines() for g in generated_diff]

    return [SequenceMatcher(None, p, g).ratio() for p, g in zip(patch_lines, generated_lines)]

def unified_diff_similarity_reward_func_filtered(patch, generated_diff, **kwargs) -> list[float]:
    """Unified diff specific reward function that removes unchanged lines and checks if the sequence of search/replace diffs matches the reference diffs."""

    assert len(patch) == len(generated_diff), "Patch and generated diff must have the same length"

    def changed_lines(diff_str: str) -> list[str]:
        lines = diff_str.splitlines()
        return [l for l in lines if l.startswith("+") or l.startswith("-")]

    patch_lines = [changed_lines(p) for p in patch]
    generated_lines = [changed_lines(g) for g in generated_diff]

    return [SequenceMatcher(None, p, g).ratio() for p, g in zip(patch_lines, generated_lines)]