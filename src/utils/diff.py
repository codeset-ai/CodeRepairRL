import re
import difflib
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod


class Diff(ABC):
    """Base class for all diff implementations."""
    
    @abstractmethod
    def parse_diff(self, diff_text: str) -> Any:
        """Parse a diff string into a structured format."""
        pass
    
    @abstractmethod
    def apply_diff(self, code: str, diff_text: str) -> str:
        """Apply a diff to the given code."""
        pass
    
    @abstractmethod
    def is_valid_format(self, diff_text: str) -> bool:
        """Validate that a diff is properly formatted."""
        pass
    
    @abstractmethod
    def extract_from_llm_response(self, response: str) -> str:
        """Extract diff blocks from an LLM response."""
        pass
    
    @abstractmethod
    def generate_diff(self, before_code: str, after_code: str) -> str:
        """Generate a diff between two code snippets."""
        pass


class SearchReplaceDiff(Diff):
    """Implementation of diff utilities using search/replace blocks format."""
    
    def parse_block(self, block: str) -> Tuple[Optional[str], Optional[str]]:
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
    
    def parse_diff(self, diff_text: str) -> List[Tuple[str, str]]:
        """
        Parse a search/replace diff into a list of (search, replace) tuples.
        
        Args:
            diff_text: A string containing one or more search/replace blocks
            
        Returns:
            A list of (search_content, replace_content) tuples
        """
        if not diff_text: return []
        
        # Split the diff into blocks
        blocks = diff_text.split("\n\n")
        result = []
        
        for block in blocks:
            search_content, replace_content = self.parse_block(block)
            if search_content is not None and replace_content is not None:
                result.append((search_content, replace_content))
        
        return result
    
    def apply_diff(self, code: str, diff_text: str) -> str:
        """
        Apply a search/replace diff to code.
        
        Args:
            code: The original code
            diff_text: The search/replace diff to apply
            
        Returns:
            The code after applying the diff
        """
        if not diff_text: return code
        
        # Parse the diff into search/replace pairs
        replacements = self.parse_diff(diff_text)
        result = code
        
        # Apply each replacement
        for search_content, replace_content in replacements:
            result = result.replace(search_content, replace_content)
        
        return result
    
    def is_valid_format(self, diff_text: str) -> bool:
        """
        Validate that a search/replace diff is properly formatted.
        
        Args:
            diff_text: The search/replace diff to validate
            
        Returns:
            True if the diff is valid, False otherwise
        """
        if not diff_text:
            return True
        
        # Split the diff into blocks
        blocks = diff_text.split("\n\n")
        
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
            search_content, replace_content = self.parse_block(block)
            if search_content is None or replace_content is None:
                return False
        
        return True
    
    def extract_from_llm_response(self, response: str) -> str:
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
    
    def generate_diff(self, before_code: str, after_code: str) -> str:
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


class UnifiedDiff(Diff):
    """Implementation of diff utilities using unified diff format."""
    
    def __init__(self, context_lines: int = 3):
        """
        Initialize a UnifiedDiff with the given number of context lines.
        
        Args:
            context_lines: Number of context lines around changes (default: 3)
        """
        self.context_lines = context_lines
    
    def parse_diff(self, diff_text: str) -> List[Dict[str, Any]]:
        """
        Parse a unified diff into a structured format.
        
        Args:
            diff_text: A string containing a unified diff
            
        Returns:
            A list of diff hunks, each containing position and change information
        """
        if not diff_text:
            return []
        
        lines = diff_text.splitlines()
        hunks = []
        current_hunk = None
        
        for line in lines:
            # Check for header or hunk lines
            if line.startswith('@@'):
                # If we have a previous hunk, add it to the result
                if current_hunk is not None:
                    hunks.append(current_hunk)
                
                # Parse the hunk header
                match = re.match(r'@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@', line)
                if match:
                    # Start a new hunk
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1
                    
                    current_hunk = {
                        'old_start': old_start,
                        'old_count': old_count,
                        'new_start': new_start,
                        'new_count': new_count,
                        'lines': []
                    }
            elif current_hunk is not None:
                # Add the line to the current hunk
                current_hunk['lines'].append(line)
        
        # Add the last hunk if there is one
        if current_hunk is not None:
            hunks.append(current_hunk)
            
        return hunks
    
    def apply_diff(self, code: str, diff_text: str) -> str:
        """
        Apply a unified diff to code.
        
        Args:
            code: The original code
            diff_text: The unified diff to apply
            
        Returns:
            The code after applying the diff
        """
        if not diff_text:
            return code
        
        # Parse the diff
        hunks = self.parse_diff(diff_text)
        if not hunks:
            return code
        
        # Split the code into lines
        lines = code.splitlines()
        result_lines = lines.copy()
        
        # Apply each hunk in reverse order to avoid line number changes
        for hunk in reversed(hunks):
            old_start = hunk['old_start'] - 1  # 0-indexed
            old_count = hunk['old_count']
            new_start = hunk['new_start'] - 1  # 0-indexed
            
            # Process the lines in the hunk
            old_lines = []
            new_lines = []
            
            for line in hunk['lines']:
                if line.startswith('-'):
                    old_lines.append(line[1:])
                elif line.startswith('+'):
                    new_lines.append(line[1:])
                elif line.startswith(' '):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
            
            # Replace the old lines with the new lines
            result_lines[old_start:old_start + old_count] = new_lines
            
        # Join lines with newline character to preserve the original format
        return '\n'.join(result_lines)
    
    def is_valid_format(self, diff_text: str) -> bool:
        """
        Validate that a unified diff is properly formatted.
        
        Args:
            diff_text: The unified diff to validate
            
        Returns:
            True if the diff is valid, False otherwise
        """
        if not diff_text:
            return True
        
        try:
            # Check if the diff contains at least one hunk header
            if not re.search(r'@@ -\d+,?\d*? \+\d+,?\d*? @@', diff_text):
                return False
                
            # Try to parse it
            hunks = self.parse_diff(diff_text)
            
            # If parsing succeeded and we got at least one hunk, it's probably valid
            return len(hunks) > 0
        except:
            return False
    
    def extract_from_llm_response(self, response: str) -> str:
        """
        Extract unified diff blocks from an LLM response.
        
        Args:
            response: The full response from an LLM
            
        Returns:
            A string containing only the unified diff blocks
        """
        # Look for blocks between triple backticks
        code_blocks = re.findall(r"```(?:diff|patch)?\n(.*?)```", response, re.DOTALL)
        
        # Filter to only include blocks that contain unified diff markers
        diff_blocks = []
        for block in code_blocks:
            if re.search(r'@@ -\d+,?\d*? \+\d+,?\d*? @@', block):
                # Normalize the block by removing any trailing newlines
                normalized_block = block.rstrip()
                diff_blocks.append(normalized_block)
        
        # Join the blocks with double newlines
        return "\n\n".join(diff_blocks)
    
    def generate_diff(self, before_code: str, after_code: str) -> str:
        """
        Generate a unified diff between two code snippets.
        
        Args:
            before_code: The original code snippet
            after_code: The fixed/modified code snippet
            
        Returns:
            A unified diff representing the changes
        """
        # Split code into lines
        before_lines = before_code.splitlines()
        after_lines = after_code.splitlines()
        
        # Use difflib to generate unified diff
        diff = difflib.unified_diff(
            before_lines, 
            after_lines,
            n=self.context_lines,
            lineterm=''
        )
        
        return '\n'.join(diff)


# Diff factory function
def get_diff(diff_type: str = 'search_replace', **kwargs) -> Diff:
    """
    Get a diff instance of the specified type.
    
    Args:
        diff_type: The type of diff to create ('search_replace' or 'unified')
        **kwargs: Additional arguments to pass to the diff constructor
        
    Returns:
        A Diff instance
    """
    if diff_type == 'search_replace':
        return SearchReplaceDiff()
    elif diff_type == 'unified':
        context_lines = kwargs.get('context_lines', 3)
        return UnifiedDiff(context_lines=context_lines)
    else:
        raise ValueError(f"Unknown diff type: {diff_type}") 