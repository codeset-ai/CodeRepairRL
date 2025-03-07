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
    def is_valid_format(self, diff_text: str, strict: bool = True) -> bool:
        """
        Validate that a diff is properly formatted.
        
        Args:
            diff_text: The diff text to validate
            strict: If True, use strict validation; if False, be more lenient
            
        Returns:
            True if the diff is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_quality(self, diff_text: str) -> float:
        """
        Assess the quality of a diff format on a scale from 0.0 to 1.0.
        
        Args:
            diff_text: The diff text to validate
            
        Returns:
            A score between 0.0 and 1.0 indicating the quality of the diff
        """
        pass
    
    @abstractmethod
    def extract_from_llm_response(self, response: str) -> str:
        """Extract diff blocks from an LLM response."""
        pass
    
    @abstractmethod
    def generate_diff(self, before_code: str, after_code: str) -> str:
        """Generate a diff between two code snippets."""
        pass
        
    def safe_apply_diff(self, code: str, diff_text: str) -> Tuple[str, float]:
        """
        Safely apply a diff to code, with quality assessment.
        If the diff has issues, it tries to recover and apply what it can.
        
        Args:
            code: The original code
            diff_text: The diff to apply
            
        Returns:
            A tuple of (resulting_code, quality_score) where quality_score
            indicates how confidently the diff was applied (1.0 = perfect)
        """
        # First check quality
        quality = self.validate_quality(diff_text)
        
        # If quality is good enough, try to apply
        if quality >= 0.4:  # Apply if at least partially recoverable
            try:
                result = self.apply_diff(code, diff_text)
                return result, quality
            except Exception as e:
                # If application fails, return original with low quality
                return code, 0.1
        
        # If quality is too low, don't attempt to apply
        return code, quality


class SearchReplaceDiff(Diff):
    """Implementation of diff utilities using search/replace blocks format."""
    
    def parse_block(self, block: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse a single search/replace block. 
        We allow the search content to be optional to facilitate diffs creating new files.
        This method is robust against common formatting errors.
        
        Args:
            block: A string containing a search/replace block
            
        Returns:
            A tuple of (search_content, replace_content), or (None, None) if parsing fails
        """
        # Try various patterns, from most exact to most forgiving
        
        # Standard pattern
        pattern_with_search = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        match = re.search(pattern_with_search, block, re.DOTALL)
        if match:
            search_content = match.group(1)
            replace_content = match.group(2)
            return search_content, replace_content
        
        # Pattern without search content (for new files)
        pattern_without_search = r"<<<<<<< SEARCH\n=======\n(.*?)\n>>>>>>> REPLACE"
        match = re.search(pattern_without_search, block, re.DOTALL)
        if match:
            return "", match.group(1)
        
        # More forgiving patterns below - these handle common LLM formatting mistakes
        
        # Slight variations in marker spacing/formatting
        pattern_forgiving = r"<+\s*S+EARCH\s*>*\n(.*?)\n=+\n(.*?)\n>+\s*R+EPLACE\s*<*"
        match = re.search(pattern_forgiving, block, re.DOTALL)
        if match:
            search_content = match.group(1)
            replace_content = match.group(2)
            return search_content, replace_content
        
        # Just get before/after with markers as separators (very forgiving)
        if "SEARCH" in block and "=====" in block and "REPLACE" in block:
            try:
                parts = re.split(r"<+[^>]*SEARCH[^>]*>+|\n=+\n|<+[^>]*REPLACE[^>]*>+", block)
                if len(parts) >= 3:  # Should have parts before, between, and after markers
                    search_content = parts[1].strip()
                    replace_content = parts[2].strip()
                    return search_content, replace_content
            except:
                pass
                
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
        
        # Try different block separators
        # First try standard double newline separator
        blocks = diff_text.split("\n\n")
        
        # If we only got one block but it contains multiple SEARCH/REPLACE markers,
        # try alternative separators
        if len(blocks) == 1 and blocks[0].count("SEARCH") > 1:
            # Try triple newline
            blocks = diff_text.split("\n\n\n")
            
            # If that didn't work, try to split on REPLACE/SEARCH boundaries
            if len(blocks) == 1 and blocks[0].count("SEARCH") > 1:
                # Look for patterns like "REPLACE ... SEARCH" which indicate block boundaries
                pattern = r"(>>>>+\s*REPLACE.*?<<<+\s*SEARCH)"
                # Split on these boundaries but keep the markers
                parts = re.split(pattern, diff_text, flags=re.DOTALL)
                
                if len(parts) > 1:
                    new_blocks = []
                    for i in range(0, len(parts), 2):
                        if i+1 < len(parts):
                            # Combine the content with the boundary
                            boundary = parts[i+1]
                            split_point = boundary.find("SEARCH")
                            if split_point != -1:
                                # Split the boundary at SEARCH
                                first_part = boundary[:split_point]
                                second_part = boundary[split_point:]
                                # Add the first block with its ending
                                new_blocks.append(parts[i] + first_part)
                                # Start the next block
                                if i+2 < len(parts):
                                    new_blocks.append(second_part + parts[i+2])
                        else:
                            new_blocks.append(parts[i])
                    
                    if len(new_blocks) > len(blocks):
                        blocks = new_blocks
                
                # If we still only have one block, try to split directly on REPLACE/SEARCH boundaries
                if len(blocks) == 1 and blocks[0].count("SEARCH") > 1:
                    # Find all occurrences of SEARCH markers
                    search_markers = [m.start() for m in re.finditer(r'<<<+\s*SEARCH', diff_text)]
                    
                    if len(search_markers) > 1:
                        new_blocks = []
                        for i in range(len(search_markers)):
                            start = search_markers[i]
                            end = search_markers[i+1] if i+1 < len(search_markers) else len(diff_text)
                            new_blocks.append(diff_text[start:end])
                        
                        blocks = new_blocks
        
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
            # Special case for empty search content - only apply if code is empty
            # This handles "new file" creation without inserting between every character
            if search_content == "":
                if result == "":
                    result = replace_content
            else:
                result = result.replace(search_content, replace_content)
        
        return result
    
    def is_valid_format(self, diff_text: str, strict: bool = True) -> bool:
        """
        Validate that a search/replace diff is properly formatted.
        
        Args:
            diff_text: The search/replace diff to validate
            strict: Whether to use strict validation (default: True)
            
        Returns:
            True if the diff is valid, False otherwise
        """
        if not diff_text:
            return True
        
        # Split the diff into blocks
        blocks = diff_text.split("\n\n")
        
        for block in blocks:
            # Check if the block contains the required markers
            if "SEARCH" not in block:
                return False
            if "=" not in block:
                return False
            if "REPLACE" not in block:
                return False
            
            if strict:
                # Strict validation - check exact markers and order
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
            
            # Parse the block to check if we can extract content
            search_content, replace_content = self.parse_block(block)
            if search_content is None or replace_content is None:
                return False
        
        return True
        
    def validate_quality(self, diff_text: str) -> float:
        """
        Assess the quality of a search/replace diff on a scale from 0.0 to 1.0.
        
        Args:
            diff_text: The search/replace diff to validate
            
        Returns:
            A score between 0.0 and 1.0 indicating the quality of the diff:
            - 1.0: Perfect format with all blocks in correct format
            - 0.7-0.9: Valid but with minor formatting issues
            - 0.4-0.6: Recoverable with non-standard format
            - 0.1-0.3: Has some diff markers but major issues
            - 0.0: Couldn't recognize as a diff at all
        """
        if not diff_text:
            return 1.0  # Empty diff is valid
            
        # Perfect format check
        if self.is_valid_format(diff_text, strict=True):
            return 1.0
            
        # Check if valid with lenient rules
        if self.is_valid_format(diff_text, strict=False):
            return 0.8  # Good enough to use but not perfect
        
        # If we get here, the diff has issues but might be partially recoverable
        
        # Split into potential blocks by double newlines
        # or try to identify block boundaries by markers
        blocks = []
        if "\n\n" in diff_text:
            blocks = diff_text.split("\n\n")
        else:
            # Try to split by diff markers
            potential_blocks = re.split(r"(<+[^>]*SEARCH|>{3,}[^<]*REPLACE)", diff_text)
            for i in range(0, len(potential_blocks)-1, 2):
                if i+1 < len(potential_blocks):
                    blocks.append(potential_blocks[i] + potential_blocks[i+1])
            
        if not blocks:
            blocks = [diff_text]  # Just treat the whole thing as one block
        
        # Calculate score based on parseable blocks
        parseable_count = 0
        total_blocks = len(blocks)
        
        for block in blocks:
            # Basic markers check
            has_search = "SEARCH" in block
            has_divider = "====" in block or "----" in block
            has_replace = "REPLACE" in block
            
            # Only counts if it has all three markers
            if has_search and has_divider and has_replace:
                search_content, replace_content = self.parse_block(block)
                if search_content is not None and replace_content is not None:
                    parseable_count += 1
                    
        if total_blocks == 0:
            return 0.0
            
        ratio = parseable_count / total_blocks
        
        # Scale from 0.1 to 0.6 based on parseable ratio
        return min(0.6, max(0.1, ratio * 0.6))
    
    def extract_from_llm_response(self, response: str) -> str:
        """
        Extract search/replace blocks from an LLM response.
        
        Args:
            response: The full response from an LLM
            
        Returns:
            A string containing only the search/replace blocks
        """
        search_replace_blocks = []
        
        # First try to find blocks between triple backticks (standard code blocks)
        code_blocks = re.findall(r"```(?:.*?)\n(.*?)```", response, re.DOTALL)
        
        for block in code_blocks:
            if "<<<<<<< SEARCH" in block and "=======" in block and ">>>>>>> REPLACE" in block:
                # Normalize the block by removing any trailing newlines
                normalized_block = block.rstrip()
                search_replace_blocks.append(normalized_block)
        
        # If no blocks found with code fences, try to extract directly
        if not search_replace_blocks:
            # Look for search/replace blocks without code fences
            direct_blocks = re.findall(
                r"<<<+\s*SEARCH\s*>*\n(.*?)\n=+\n(.*?)\n>+\s*REPLACE\s*<*", 
                response, 
                re.DOTALL
            )
            
            for search_content, replace_content in direct_blocks:
                block = (
                    "<<<<<<< SEARCH\n"
                    f"{search_content}\n"
                    "=======\n"
                    f"{replace_content}\n"
                    ">>>>>>> REPLACE"
                )
                search_replace_blocks.append(block)
        
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
        This method is robust against common formatting errors in LLM outputs.
        
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
            # Check for header or hunk lines - with robust matching
            if '@' in line and '-' in line and '+' in line:
                # Try to parse as a hunk header
                
                # If we have a previous hunk, add it to the result
                if current_hunk is not None:
                    # Validate the hunk before adding it
                    if self._validate_hunk(current_hunk):
                        hunks.append(current_hunk)
                
                # First try standard format
                match = re.search(r'@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@', line)
                
                # If no match, try more forgiving pattern
                if not match:
                    match = re.search(r'@+\s*-\s*(\d+)[,:]?(\d+)?\s+\+\s*(\d+)[,:]?(\d+)?', line)
                
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
                else:
                    # If we can't parse, just create a best-guess hunk
                    # This allows recovery from malformed header but valid content
                    nums = re.findall(r'\d+', line)
                    if len(nums) >= 2:
                        try:
                            current_hunk = {
                                'old_start': int(nums[0]),
                                'old_count': int(nums[1]) if len(nums) > 2 else 1,
                                'new_start': int(nums[2]) if len(nums) > 2 else int(nums[1]),
                                'new_count': int(nums[3]) if len(nums) > 3 else 1,
                                'lines': []
                            }
                        except (ValueError, IndexError):
                            # If conversion fails, create a minimal valid hunk
                            current_hunk = {
                                'old_start': 1,
                                'old_count': 1,
                                'new_start': 1,
                                'new_count': 1,
                                'lines': []
                            }
            elif current_hunk is not None:
                # Add the line to the current hunk
                # Normalize leading characters if needed
                if line and line[0] not in ['+', '-', ' ']:
                    # Try to guess the line type
                    if line.startswith('add') or line.startswith('ins'):
                        line = '+' + line[3:]
                    elif line.startswith('del') or line.startswith('rem'):
                        line = '-' + line[3:]
                    else:
                        # Default to context line
                        line = ' ' + line
                
                current_hunk['lines'].append(line)
        
        # Add the last hunk if there is one
        if current_hunk is not None:
            # Validate the hunk before adding it
            if self._validate_hunk(current_hunk):
                hunks.append(current_hunk)
            
        return hunks
    
    def _validate_hunk(self, hunk: Dict[str, Any]) -> bool:
        """
        Validate a hunk to ensure it has valid line counts and content.
        
        Args:
            hunk: A hunk dictionary to validate
            
        Returns:
            True if the hunk is valid, False otherwise
        """
        # Check if the hunk has any lines
        if not hunk['lines']:
            return False
            
        # Check if line counts are positive
        if hunk['old_count'] <= 0 or hunk['new_count'] <= 0:
            return False
            
        # Check if start positions are positive
        if hunk['old_start'] <= 0 or hunk['new_start'] <= 0:
            return False
            
        return True
    
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
    
    def is_valid_format(self, diff_text: str, strict: bool = True) -> bool:
        """
        Validate that a unified diff is properly formatted.
        
        Args:
            diff_text: The unified diff to validate
            strict: Whether to use strict validation (default: True)
            
        Returns:
            True if the diff is valid, False otherwise
        """
        if not diff_text:
            return True
        
        try:
            if strict:
                # Check if the diff contains at least one properly formatted hunk header
                if not re.search(r'@@ -\d+,?\d*? \+\d+,?\d*? @@', diff_text):
                    return False
            else:
                # More lenient check - just look for something that resembles a hunk header
                if not re.search(r'@+\s*-\s*\d+.*\+\s*\d+.*@+', diff_text):
                    return False
                
            # Try to parse it
            hunks = self.parse_diff(diff_text)
            
            # If parsing succeeded and we got at least one hunk, it's probably valid
            return len(hunks) > 0
        except:
            return False
            
    def validate_quality(self, diff_text: str) -> float:
        """
        Assess the quality of a unified diff on a scale from 0.0 to 1.0.
        
        Args:
            diff_text: The unified diff to validate
            
        Returns:
            A score between 0.0 and 1.0 indicating the quality of the diff:
            - 1.0: Perfect format with all hunks properly formatted
            - 0.7-0.9: Valid but with minor formatting issues
            - 0.4-0.6: Recoverable with non-standard format
            - 0.1-0.3: Has some diff markers but major issues
            - 0.0: Couldn't recognize as a diff at all
        """
        if not diff_text:
            return 1.0  # Empty diff is valid
            
        # Perfect format check
        if self.is_valid_format(diff_text, strict=True):
            return 1.0
            
        # Check if valid with lenient rules
        if self.is_valid_format(diff_text, strict=False):
            return 0.8  # Good enough to use but not perfect
        
        # Count basic indicators of diff format
        has_hunk_markers = '@' in diff_text
        has_add_lines = re.search(r'^\+', diff_text, re.MULTILINE) is not None
        has_remove_lines = re.search(r'^-', diff_text, re.MULTILINE) is not None
        has_context_lines = re.search(r'^ ', diff_text, re.MULTILINE) is not None
        
        # Score based on markers present
        score = 0.0
        if has_hunk_markers:
            score += 0.2
        if has_add_lines:
            score += 0.1
        if has_remove_lines:
            score += 0.1
        if has_context_lines:
            score += 0.1
            
        # Check if we can recover any hunks
        try:
            # Try to recognize hunk headers with a very forgiving pattern
            potential_hunks = re.split(r'(?:^|\n)@+[^@\n]*@+', diff_text)
            
            # If we found potential hunks, check if any looks valid
            for hunk in potential_hunks[1:]:  # Skip first part (before first header)
                has_valid_content = False
                lines = hunk.splitlines()
                for line in lines:
                    # Check if line starts with diff markers
                    if line and line[0] in ['+', '-', ' ']:
                        has_valid_content = True
                        break
                        
                if has_valid_content:
                    score += 0.1  # Add a bit for each recoverable hunk
            
            score = min(0.6, score)  # Cap at 0.6 for partial recovery
        except:
            # If analysis fails, just return the basic marker score
            pass
            
        return score
    
    def extract_from_llm_response(self, response: str) -> str:
        """
        Extract unified diff blocks from an LLM response.
        
        Args:
            response: The full response from an LLM
            
        Returns:
            A string containing only the unified diff blocks
        """
        diff_blocks = []
        
        # First try to find blocks between triple backticks
        code_blocks = re.findall(r"```(?:diff|patch)?\n(.*?)```", response, re.DOTALL)
        
        for block in code_blocks:
            if re.search(r'@@ -\d+,?\d*? \+\d+,?\d*? @@', block):
                # Normalize the block by removing any trailing newlines
                normalized_block = block.rstrip()
                diff_blocks.append(normalized_block)
        
        # If no blocks found with code fences, try to extract directly
        if not diff_blocks:
            # Look for hunk headers directly in the text
            hunk_headers = re.findall(r'@@ -\d+,?\d*? \+\d+,?\d*? @@', response)
            
            if hunk_headers:
                # Split the response by hunk headers
                parts = re.split(r'(@@ -\d+,?\d*? \+\d+,?\d*? @@)', response)
                
                # Reconstruct the diff blocks
                for i in range(1, len(parts), 2):
                    if i+1 < len(parts):
                        # Combine header with content
                        hunk = parts[i] + parts[i+1]
                        # Extract lines until we hit something that's clearly not part of the diff
                        lines = []
                        for line in hunk.splitlines():
                            if line.startswith('+') or line.startswith('-') or line.startswith(' '):
                                lines.append(line)
                            elif re.match(r'@@ -\d+,?\d*? \+\d+,?\d*? @@', line):
                                lines.append(line)
                            else:
                                # Stop at non-diff content
                                break
                        
                        if lines:
                            diff_blocks.append('\n'.join(lines))
        
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