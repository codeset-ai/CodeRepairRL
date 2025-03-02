import re
import difflib
from typing import List, Tuple, Optional


def parse_search_replace_block(block: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a single search/replace block. 
    We allow the search content to be optional to facilitate diffs creating new files
    Args:
        block: A string containing a search/replace block
        
    Returns:
        A tuple of (search_content, replace_content), or (None, None) if parsing fails
    """
    pattern_with_search = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    
    # Use re.DOTALL to match across multiple lines
    match = re.search(pattern_with_search, block, re.DOTALL)
    
    if match:
        search_content = match.group(1)
        replace_content = match.group(2)
        return search_content, replace_content
    
    pattern_without_search = r"<<<<<<< SEARCH\n=======\n(.*?)\n>>>>>>> REPLACE"

    match = re.search(pattern_without_search, block, re.DOTALL)

    if match:
        return "", match.group(1)
    
    return None, None


def parse_search_replace_diff(diff: str) -> List[Tuple[str, str]]:
    """
    Parse a search/replace diff into a list of (search, replace) tuples.
    
    Args:
        diff: A string containing one or more search/replace blocks
        
    Returns:
        A list of (search_content, replace_content) tuples
    """
    if not diff: return []
    
    # Split the diff into blocks
    blocks = diff.split("\n\n")
    result = []
    
    for block in blocks:
        search_content, replace_content = parse_search_replace_block(block)
        if search_content is not None and replace_content is not None:
            result.append((search_content, replace_content))
    
    return result


def apply_search_replace_diff(code: str, diff: str) -> str:
    """
    Apply a search/replace diff to code.
    
    Args:
        code: The original code
        diff: The search/replace diff to apply
        
    Returns:
        The code after applying the diff
    """
    if not diff: return code
    
    # Parse the diff into search/replace pairs
    replacements = parse_search_replace_diff(diff)
    result = code
    
    # Apply each replacement
    for search_content, replace_content in replacements:
        result = result.replace(search_content, replace_content)
    
    return result


def is_valid_diff_format(diff: str) -> bool:
    """
    Validate that a search/replace diff is properly formatted.
    
    Args:
        diff: The search/replace diff to validate
        
    Returns:
        True if the diff is valid, False otherwise
    """
    if not diff:
        return True
    
    # Split the diff into blocks
    blocks = diff.split("\n\n")
    
    for block in blocks:
        # Check if the block contains the required markers
        if "<<<<<<< SEARCH" not in block:
            return False
        if "=======" not in block:
            return False
        if ">>>>>>> REPLACE" not in block:
            return False
        
        # Check if the markers are in the correct order
        search_idx = block.find("<<<<<<< SEARCH")
        divider_idx = block.find("=======")
        replace_idx = block.find(">>>>>>> REPLACE")
        
        if not (search_idx < divider_idx < replace_idx):
            return False
        
        # Parse the block to ensure it's valid
        search_content, replace_content = parse_search_replace_block(block)
        if search_content is None or replace_content is None:
            return False
    
    return True


def extract_search_replace_blocks_from_llm_response(response: str) -> str:
    """
    Extract search/replace blocks from an LLM response.
    
    Args:
        response: The full response from an LLM
        
    Returns:
        A string containing only the search/replace blocks
    """
    # Look for blocks between triple backticks
    code_blocks = re.findall(r"```(?:.*?)\n(.*?)```", response, re.DOTALL)
    
    # Filter to only include blocks that contain search/replace markers
    search_replace_blocks = []
    for block in code_blocks:
        if "<<<<<<< SEARCH" in block and "=======" in block and ">>>>>>> REPLACE" in block:
            # Normalize the block by removing any trailing newlines
            normalized_block = block.rstrip()
            search_replace_blocks.append(normalized_block)
    
    # Join the blocks with double newlines
    return "\n\n".join(search_replace_blocks)


def generate_search_replace_diff(before_code: str, after_code: str) -> str:
    """
    Generate a SEARCH/REPLACE diff between before and after code versions.
    
    Args:
        before_code: The original code snippet
        after_code: The fixed/modified code snippet
        
    Returns:
        A SEARCH/REPLACE diff representing the changes, focusing on changed chunks
    """
    # Split code into lines
    before_lines = before_code.splitlines()
    after_lines = after_code.splitlines()
    
    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    search_replace_blocks = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # We only care about changes for SEARCH/REPLACE format
        if tag == 'replace':
            search_chunk = '\n'.join(before_lines[i1:i2])
            replace_chunk = '\n'.join(after_lines[j1:j2])
            
            # Create a SEARCH/REPLACE block
            block = "<<<<<<< SEARCH\n"
            block += f"{search_chunk}\n"
            block += "=======\n"
            block += f"{replace_chunk}\n"
            block += ">>>>>>> REPLACE"
            
            search_replace_blocks.append(block)
        
        # For deletions, we just need a SEARCH block with empty REPLACE
        elif tag == 'delete':
            search_chunk = '\n'.join(before_lines[i1:i2])
            
            block = "<<<<<<< SEARCH\n"
            block += f"{search_chunk}\n"
            block += "=======\n"
            block += ">>>>>>> REPLACE"
            
            search_replace_blocks.append(block)
        
        # For insertions, we need context (the line before insertion)
        elif tag == 'insert':
            replace_chunk = '\n'.join(after_lines[j1:j2])
            
            # Need to find the context (the line before insertion)
            context_line = ""
            if i1 > 0:
                context_line = before_lines[i1-1]
            
            block = "<<<<<<< SEARCH\n"
            if context_line:
                block += f"{context_line}\n"
            block += "=======\n"
            if context_line:
                block += f"{context_line}\n"
            block += f"{replace_chunk}\n"
            block += ">>>>>>> REPLACE"
            
            search_replace_blocks.append(block)
    
    # Join all blocks
    return "\n\n".join(search_replace_blocks) 