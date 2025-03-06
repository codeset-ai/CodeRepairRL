import re
import difflib

from src.utils.diff import get_diff


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


def count_search_replace_markers(text) -> float:
    """
    Calculate a partial reward for search/replace diff format based on presence of markers.
    
    Args:
        text: The text to analyze for diff markers
        
    Returns:
        A score between 0.0 and 0.3 indicating how well the text follows diff format
    """    
    score = 0.0
    
    if "SEARCH" in text:
        score += 0.1
    if "=======" in text:
        score += 0.1
    if "REPLACE" in text:
        score += 0.1
    
    return score

def count_unified_diff_markers(text) -> float:
    """
    Calculate a partial reward for unified diff format based on presence of markers.
    
    Args:
        text: The text to analyze for unified diff markers
        
    Returns:
        A score between 0.0 and 0.4 indicating how well the text follows unified diff format
    """
    score = 0.0
    
    if "@@ " in text:
        score += 0.1
    if " @@" in text:
        score += 0.1
    
    # Check for line markers
    if re.search(r'^\+', text, re.MULTILINE):
        score += 0.1
    if re.search(r'^-', text, re.MULTILINE):
        score += 0.1
        
    return score

def partial_diff_format_reward_func(completions, diff_type="search_replace", **kwargs) -> list[float]:
    """
    Reward function that gives partial credit for diff format.
    Takes no account of the number of blocks nor ordering.
    
    Args:
        completions: List of model completions
        diff_type: Type of diff to check ("search_replace" or "unified")
        
    Returns:
        List of scores between 0.0 and 0.4 for each completion
    """
    contents = [completion[0]["content"] for completion in completions]
    
    if diff_type == "search_replace":
        return [count_search_replace_markers(c) for c in contents]
    elif diff_type == "unified":
        return [count_unified_diff_markers(c) for c in contents]
    else:
        raise ValueError(f"Unknown diff type: {diff_type}")
        
def diff_quality_reward_func(completions, diff_type="search_replace", **kwargs) -> list[float]:
    """
    Reward function that evaluates the quality of diffs using the quality metric.
    
    Args:
        completions: List of model completions
        diff_type: Type of diff to check ("search_replace" or "unified")
        
    Returns:
        List of quality scores between 0.0 and 1.0 for each completion
    """
    # Create diff instance based on type
    diff = get_diff(diff_type)
    
    # Extract content
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_xml_answer(c) for c in contents]
    extracted_diffs = [diff.extract_from_llm_response(a) for a in answers]
    
    # Calculate quality scores
    return [diff.validate_quality(d) for d in extracted_diffs]

def strict_diff_format_reward_func(completions, diff_type="search_replace", **kwargs) -> list[float]:
    """
    Reward function that gives full credit for diff format.
    
    Args:
        completions: List of model completions
        diff_type: Type of diff to check ("search_replace" or "unified")
        
    Returns:
        List of binary scores (1.0 or 0.0) indicating if each completion has valid format
    """
    # Create diff instance based on type
    diff = get_diff(diff_type)
    
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_xml_answer(c) for c in contents]
    blocks = [diff.extract_from_llm_response(a) for a in answers]
    return [1.0 if diff.is_valid_format(b, strict=True) else 0.0 for b in blocks]

def diff_similarity_reward(reference_blocks: str, generated_blocks: str, diff_type="search_replace") -> float:
    """
    Calculate similarity between two diffs using SequenceMatcher.
    Also checks if the generated diff is in valid format.
    
    Args:
        reference_blocks: The ground truth diff blocks
        generated_blocks: The generated diff blocks
        diff_type: Type of diff to check ("search_replace" or "unified")
    
    Returns:
        A similarity score between 0.0 and 1.0, where:
          1.0 means identical diffs
          Values between 0.0-1.0 represent partial similarity
          0.0 means invalid diff format
    """
    # Create diff instance based on type
    diff = get_diff(diff_type)
    
    # Check quality and adjust score based on it
    quality = diff.validate_quality(generated_blocks)
    
    # If quality is too low, return a low score
    if quality < 0.4:
        return quality * 0.5  # Scale quality to be max 0.2 for very low quality
    
    # Calculate similarity based on diff type
    if diff_type == "search_replace":
        # Process each block separately
        ref_blocks = reference_blocks.split("\n\n")
        gen_blocks = generated_blocks.split("\n\n")
        
        # Handle different number of blocks
        if len(ref_blocks) != len(gen_blocks):
            # Penalize for different number of blocks but still calculate partial similarity
            block_count_penalty = 0.7
        else:
            block_count_penalty = 1.0
            
        # Calculate similarity for available blocks
        min_len = min(len(ref_blocks), len(gen_blocks))
        if min_len == 0:
            return 0.0
            
        similarities = []
        for i in range(min_len):
            ref_block = ref_blocks[i]
            gen_block = gen_blocks[i]
            ref_lines = ref_block.splitlines()
            gen_lines = gen_block.splitlines()
            seq_matcher = difflib.SequenceMatcher(None, ref_lines, gen_lines)
            similarity = seq_matcher.ratio()
            similarities.append(similarity)
        
        # Average similarity multiplied by the block count penalty
        return (sum(similarities) / len(similarities)) * block_count_penalty * (quality / 1.0)
    
    elif diff_type == "unified":
        # For unified diffs, compare line by line
        ref_lines = reference_blocks.splitlines()
        gen_lines = generated_blocks.splitlines()
        
        # Remove metadata lines (like --- and +++ lines)
        ref_lines = [l for l in ref_lines if not l.startswith('---') and not l.startswith('+++')]
        gen_lines = [l for l in gen_lines if not l.startswith('---') and not l.startswith('+++')]
        
        # Calculate line-by-line similarity
        seq_matcher = difflib.SequenceMatcher(None, ref_lines, gen_lines)
        similarity = seq_matcher.ratio()
        
        return similarity * (quality / 1.0)
    
    else:
        raise ValueError(f"Unknown diff type: {diff_type}")

def diff_similarity_reward_func(completions, reference, diff_type="search_replace", **kwargs) -> list[float]:
    """
    Reward function that compares generated diffs with reference diffs using sequence matching.
    
    Args:
        completions: List of model completions
        reference: List of reference diff blocks to compare against
        diff_type: Type of diff to check ("search_replace" or "unified")
        
    Returns:
        List of similarity scores between 0 and 1 for each completion
    """
    # Create diff instance based on type
    diff = get_diff(diff_type)
    
    # Extract content from completions
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_xml_answer(c) for c in contents]
    generated = [diff.extract_from_llm_response(a) for a in answers]
    
    # Calculate similarity for each completion-reference pair
    similarities = [
        diff_similarity_reward(ref_blocks, gen_blocks, diff_type=diff_type) 
        for ref_blocks, gen_blocks in zip(reference, generated)
    ]
    
    return similarities