from src.utils.diff import SearchReplaceDiff
from src.utils.extract import extract_xml_answer, extract_markdown_block

# Diff specific reward functions

def diff_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks the quality of the extracted diff format between 0.0 and 1.0."""
    
    answer_contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    markdown_contents = [extract_markdown_block(answer_content) for answer_content in answer_contents]
    
    # Use the static method directly on the markdown_contents to evaluate format quality
    return [SearchReplaceDiff.validate_quality(diff_str) for diff_str in markdown_contents]

def diff_similarity_reward_func(completions, diffs, **kwargs) -> list[float]:
    """Reward function that checks if the sequence of search/replace diffs matches the reference diffs."""
    
    answer_contents = [extract_xml_answer(completion[0]["content"]) for completion in completions] # extract contents of <answer> tags
    markdown_contents = [extract_markdown_block(answer_content) for answer_content in answer_contents]
    generated_diffs_batch = [SearchReplaceDiff.from_string(diff_str) for diff_str in markdown_contents]
    reference_diffs_batch = [SearchReplaceDiff.from_string(diff_str) for diff_str in diffs]
    
    return [diff.similarity(ref_diff) for diff, ref_diff in zip(generated_diffs_batch, reference_diffs_batch)]