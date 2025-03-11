import re
import difflib
from typing import List, Tuple, Optional, Dict, Any, Type, TypeVar, ClassVar
from abc import ABC, abstractmethod

T = TypeVar('T', bound='Diff')

class Diff(ABC):
    """
    Base class for all diff implementations.
    
    Diff implementations are designed to be robust against malformed or poorly formatted inputs,
    attempting to parse and apply diffs even when they don't perfectly match the expected format.
    Each implementation defines its own tolerance for format errors and recovery strategies.
    """

    @staticmethod
    @abstractmethod
    def extract_from_llm_response(response: str) -> List[T]:
        """Extract diff blocks from an LLM response and return a list of Diff objects."""
        pass
    
    @classmethod
    @abstractmethod
    def from_string(cls: Type[T], diff_text: str) -> T:
        """Parse a diff string into a structured Diff object."""
        pass

    @classmethod
    @abstractmethod
    def from_codes(cls: Type[T], before_code: str, after_code: str) -> T:
        """Generate a Diff object representing the changes between two code snippets."""
        pass
    
    @abstractmethod
    def apply_diff(self, code: str) -> str:
        """Apply this diff to the given code."""
        pass
    
    @abstractmethod
    def is_valid_format(self, strict: bool = True) -> bool:
        """
        Validate that this diff is properly formatted.
        
        Args:
            strict: If True, use strict validation; if False, be more lenient
            
        Returns:
            True if the diff is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_quality(self) -> float:
        """
        Assess the quality of this diff format on a scale from 0.0 to 1.0.
        
        Returns:
            A score between 0.0 and 1.0 indicating the quality of the diff, 1.0 is perfect
        """
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert this diff object to its string representation."""
        pass
        
    def safe_apply_diff(self, code: str) -> Tuple[str, float]:
        """
        Safely apply this diff to code, with quality assessment.
        
        Evaluates the quality of the diff format before applying it.
        Returns both the modified code and a quality score indicating
        how well-formed the diff was.
        
        Args:
            code: The original code
            
        Returns:
            A tuple of (modified_code, quality_score)
        """
        # First check quality
        quality = self.validate_quality()
        
        # If quality is good enough, try to apply
        if quality >= 0.4:  # Apply if at least partially recoverable
            try:
                result = self.apply_diff(code)
                return result, quality
            except Exception as e:
                # If application fails, return original with low quality
                return code, 0.1
        
        # If quality is too low, don't attempt to apply
        return code, quality


class SearchReplaceDiff(Diff):
    """
    Implementation of diff utilities using search/replace blocks format.
    
    Robust against common formatting errors in LLM outputs, including:
    - Variations in marker syntax (e.g., different numbers of < or > characters)
    - Whitespace variations around markers
    - Missing or malformed block separators
    - Blocks without code fences in LLM responses
    """
    
    def __init__(self, blocks: List[Tuple[str, str]]):
        """
        Initialize with a list of (search, replace) tuples.
        
        Args:
            blocks: List of (search_content, replace_content) tuples
        """
        self.blocks = blocks
    
    @classmethod
    def from_string(cls, diff_text: str) -> 'SearchReplaceDiff':
        """
        Parse a search/replace diff into a SearchReplaceDiff object.
        
        Handles various block separator formats and is robust against common LLM formatting errors.
        
        Args:
            diff_text: A string containing one or more search/replace blocks
            
        Returns:
            A SearchReplaceDiff object
        """
        if not diff_text: 
            return cls([])
        
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
                    blocks = []
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
                                blocks.append(parts[i] + first_part)
                                # Start the next block
                                if i+2 < len(parts):
                                    blocks.append(second_part + parts[i+2])
                        else:
                            blocks.append(parts[i])
        
        result = []
        
        for block in blocks:
            # Try various patterns, from most exact to most forgiving
            
            # Standard pattern
            pattern_with_search = r"<<<+\s*SEARCH\s*>*\n(.*?)\n=+\n(.*?)\n>>>+\s*REPLACE\s*<*"
            match = re.search(pattern_with_search, block, re.DOTALL)
            if match:
                search_content = match.group(1)
                replace_content = match.group(2)
                result.append((search_content, replace_content))
                continue
            
            # Pattern without search content (for new files)
            pattern_without_search = r"<<<+\s*SEARCH\s*>*\n=+\n(.*?)\n>>>+\s*REPLACE\s*<*"
            match = re.search(pattern_without_search, block, re.DOTALL)
            if match:
                result.append(("", match.group(1)))
                continue
            
            # Just get before/after with markers as separators (very forgiving)
            if "SEARCH" in block and "=====" in block and "REPLACE" in block:
                try:
                    parts = re.split(r"<+[^>]*SEARCH[^>]*>+|\n=+\n|<+[^>]*REPLACE[^>]*>+", block)
                    if len(parts) >= 3:  # Should have parts before, between, and after markers
                        search_content = parts[1].strip()
                        replace_content = parts[2].strip()
                        result.append((search_content, replace_content))
                except:
                    pass
        
        return cls(result)
    
    @classmethod
    def from_codes(cls, before_code: str, after_code: str) -> 'SearchReplaceDiff':
        """
        Generate a SearchReplaceDiff object representing the changes between before and after code versions.
        
        Uses difflib to intelligently find the changes and create search/replace blocks with context.
        
        Args:
            before_code: The original code snippet
            after_code: The fixed/modified code snippet
            
        Returns:
            A SearchReplaceDiff object representing the changes
        """
        if before_code == after_code:
            return cls([])
        
        # Split code into lines
        before_lines = before_code.splitlines()
        after_lines = after_code.splitlines()
        
        # Use SequenceMatcher to find differences
        matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
        blocks = []
        
        # Context lines to include before and after changes
        context_lines = 2
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
                
            # Calculate context boundaries
            start1 = max(0, i1 - context_lines)
            end1 = min(len(before_lines), i2 + context_lines)
            start2 = max(0, j1 - context_lines)
            end2 = min(len(after_lines), j2 + context_lines)
            
            # Extract the changed lines with context
            before_chunk = '\n'.join(before_lines[start1:end1])
            after_chunk = '\n'.join(after_lines[start2:end2])
            
            # Only add non-empty chunks
            if before_chunk or after_chunk:
                blocks.append((before_chunk, after_chunk))
        
        # If no specific changes were found but the files are different,
        # fall back to treating the entire files as a single change
        if not blocks and before_code != after_code:
            blocks = [(before_code, after_code)]
            
        return cls(blocks)
    
    @staticmethod
    def extract_from_llm_response(response: str) -> List['SearchReplaceDiff']:
        """
        Extract search/replace blocks from an LLM response and return a list of SearchReplaceDiff objects.
        
        Args:
            response: The full response from an LLM
            
        Returns:
            A list of SearchReplaceDiff objects
        """
        # First try to find blocks between triple backticks
        code_blocks = re.findall(r"```(?:.*?)\n(.*?)```", response, re.DOTALL)
        
        # If no blocks found with code fences, try to extract directly
        if not code_blocks:
            code_blocks = [response]
            
        return [SearchReplaceDiff.from_string(block) for block in code_blocks if block.strip()]
    
    def apply_diff(self, code: str) -> str:
        """
        Apply this search/replace diff to code.
        
        Args:
            code: The original code
            
        Returns:
            The code after applying the diff
        """
        if not self.blocks: 
            return code
        
        result = code
        
        # Apply each search/replace pair in sequence
        for search_content, replace_content in self.blocks:
            if not search_content:
                # If search is empty, this is a new file creation
                if not result:
                    result = replace_content
            else:
                # Otherwise, perform the replacement
                result = result.replace(search_content, replace_content)
        
        return result
    
    def validate_quality(self) -> float:
        """
        Assess the quality of this diff format on a scale from 0.0 to 1.0.
        
        Returns:
            A score between 0.0 and 1.0 indicating the quality of the diff
        """
        if not self.blocks:
            return 0.0
        
        # Start with a perfect score
        score = 1.0
        
        # Check each block for quality issues
        for search_content, replace_content in self.blocks:
            # Both parts should exist (though search can be empty for new files)
            if replace_content is None:
                score -= 0.3
                continue
                
            # Check that the replacement isn't identical to the search
            if search_content == replace_content and search_content:
                score -= 0.2
                
            # Empty blocks are invalid
            if not replace_content and not search_content:
                score = 0.0
                break
                
            # Penalize very large blocks slightly (they're more error-prone)
            if search_content and len(search_content) > 1000:
                score -= 0.1
                
            # Penalize very small blocks slightly (they might be too granular)
            if search_content and len(search_content) < 3:
                score -= 0.1
        
        # Normalize score to 0.0-1.0 range
        return min(1.0, max(0.0, score))
    
    def is_valid_format(self, strict: bool = True) -> bool:
        """
        Validate that this diff is properly formatted.
        
        Args:
            strict: If True, use strict validation; if False, be more lenient
            
        Returns:
            True if the diff is valid, False otherwise
        """
        return self.validate_quality() == 1.0 if strict else self.validate_quality() >= 0.4
    
    
    def to_string(self) -> str:
        """
        Convert this diff object to its string representation.
        
        Returns:
            A string containing the search/replace blocks
        """
        search_replace_blocks = []
        
        for search_content, replace_content in self.blocks:
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


class UnifiedDiff(Diff):
    """
    Implementation of diff utilities using unified diff format.
    
    Unified diffs are a standard format used by tools like git diff.
    This implementation is robust against common formatting errors in LLM outputs.
    """
    
    def __init__(self, hunks: List[Dict[str, Any]], context_lines: int = 3):
        """
        Initialize with a list of hunks and context line count.
        
        Args:
            hunks: List of hunk dictionaries
            context_lines: Number of context lines to include in diffs (default: 3)
        """
        self.hunks = hunks
        self.context_lines = context_lines
    
    @classmethod
    def from_string(cls, diff_text: str) -> 'UnifiedDiff':
        """
        Parse a unified diff string into a UnifiedDiff object.
        
        Handles standard unified diff format with @@ markers and +/- line prefixes.
        Attempts to recover from common formatting errors.
        
        Args:
            diff_text: A string containing a unified diff
            
        Returns:
            A UnifiedDiff object
        """
        if not diff_text:
            return cls([], 3)
            
        hunks = []
        current_hunk = None
        
        for line in diff_text.splitlines():
            # Look for hunk headers
            hunk_match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)', line)
            
            if hunk_match:
                # If we were processing a hunk, add it to the list
                if current_hunk is not None:
                    hunks.append(current_hunk)
                
                # Parse the hunk header
                start1 = int(hunk_match.group(1))
                count1 = int(hunk_match.group(2) or 1)
                start2 = int(hunk_match.group(3))
                count2 = int(hunk_match.group(4) or 1)
                heading = hunk_match.group(5).strip()
                
                # Create a new hunk
                current_hunk = {
                    'start1': start1,
                    'count1': count1,
                    'start2': start2,
                    'count2': count2,
                    'heading': heading,
                    'lines': []
                }
            elif current_hunk is not None:
                # Add the line to the current hunk
                if line.startswith('+') or line.startswith('-') or line.startswith(' '):
                    current_hunk['lines'].append(line)
                # Handle other lines that might be part of the diff but not properly formatted
                elif line.strip() and not line.startswith('diff ') and not line.startswith('index '):
                    # Try to recover malformed lines by guessing their type
                    if line.startswith('+++') or line.startswith('---'):
                        # These are file headers, ignore them
                        continue
                    else:
                        # Assume it's a context line if we can't tell
                        current_hunk['lines'].append(' ' + line)
        
        # Add the last hunk if there is one
        if current_hunk is not None:
            hunks.append(current_hunk)
            
        # Determine context lines from the diff if possible
        context_lines = 3  # Default
        if hunks:
            # Try to infer context lines from the diff
            context_line_counts = []
            for hunk in hunks:
                # Count consecutive context lines at the beginning of the hunk
                count = 0
                for line in hunk['lines']:
                    if line.startswith(' '):
                        count += 1
                    else:
                        break
                if count > 0:
                    context_line_counts.append(count)
            
            if context_line_counts:
                # Use the most common context line count
                context_lines = max(set(context_line_counts), key=context_line_counts.count)
        
        return cls(hunks, context_lines)
    
    def _validate_hunk(self, hunk: Dict[str, Any]) -> bool:
        """
        Validate that a hunk is properly formatted.
        
        Args:
            hunk: The hunk to validate
            
        Returns:
            True if the hunk is valid, False otherwise
        """
        # Check that the hunk has all required fields
        required_fields = ['start1', 'count1', 'start2', 'count2', 'lines']
        for field in required_fields:
            if field not in hunk:
                return False
                
        # Check that line counts match the actual lines
        plus_count = sum(1 for line in hunk['lines'] if line.startswith('+'))
        minus_count = sum(1 for line in hunk['lines'] if line.startswith('-'))
        
        # In a valid hunk, the line counts should match the header
        # But we allow some flexibility for malformed diffs
        if abs(plus_count - hunk['count2']) > 2 or abs(minus_count - hunk['count1']) > 2:
            return False
            
        return True
    
    def apply_diff(self, code: str) -> str:
        """
        Apply this unified diff to code.
        
        Args:
            code: The original code
            
        Returns:
            The code after applying the diff
        """
        if not self.hunks:
            return code
            
        # Split the code into lines
        lines = code.splitlines()
        result_lines = lines.copy()
        
        # Apply each hunk in reverse order to avoid line number changes
        for hunk in reversed(self.hunks):
            start1 = hunk['start1']
            count1 = hunk['count1']
            start2 = hunk['start2']
            count2 = hunk['count2']
            
            # Extract the lines from the hunk
            hunk_lines = hunk['lines']
            
            # Verify that the hunk matches the code
            original_lines = lines[start1-1:start1-1+count1]
            hunk_original = [line[1:] for line in hunk_lines if line.startswith(' ') or line.startswith('-')]
            
            # Check if the hunk matches the code
            # We use a fuzzy match to handle minor differences
            if len(original_lines) != len(hunk_original):
                # Try to find the closest match
                for offset in range(-5, 6):  # Try offsets from -5 to +5
                    if start1 + offset - 1 < 0:
                        continue
                    if start1 + offset - 1 + count1 > len(lines):
                        continue
                        
                    test_lines = lines[start1+offset-1:start1+offset-1+count1]
                    if len(test_lines) == len(hunk_original):
                        # Check similarity
                        similarity = sum(1 for a, b in zip(test_lines, hunk_original) if a == b) / len(test_lines)
                        if similarity > 0.7:  # If more than 70% match, use this offset
                            start1 += offset
                            break
            
            # Extract the new lines from the hunk
            new_lines = [line[1:] for line in hunk_lines if line.startswith(' ') or line.startswith('+')]
            
            # Replace the lines in the result
            result_lines[start1-1:start1-1+count1] = new_lines
        
        # Join the lines back into a string
        return '\n'.join(result_lines)
    
    def is_valid_format(self, strict: bool = True) -> bool:
        """
        Validate that this diff is properly formatted.
        
        Args:
            strict: If True, use strict validation; if False, be more lenient
            
        Returns:
            True if the diff is valid, False otherwise
        """
        if not self.hunks:
            return True
            
        # Check each hunk
        for hunk in self.hunks:
            if strict:
                # In strict mode, validate each hunk thoroughly
                if not self._validate_hunk(hunk):
                    return False
            else:
                # In lenient mode, just check that the hunk has lines
                if 'lines' not in hunk or not hunk['lines']:
                    return False
        
        return True
    
    def validate_quality(self) -> float:
        """
        Assess the quality of this diff format on a scale from 0.0 to 1.0.
        
        Returns:
            A score between 0.0 and 1.0 indicating the quality of the diff
        """
        if not self.hunks:
            return 0.0
            
        # Start with a perfect score
        score = 1.0
        
        # Check each hunk for quality issues
        for hunk in self.hunks:
            # Check that the hunk is valid
            if not self._validate_hunk(hunk):
                score -= 0.3
                continue
                
            # Check line counts
            plus_count = sum(1 for line in hunk['lines'] if line.startswith('+'))
            minus_count = sum(1 for line in hunk['lines'] if line.startswith('-'))
            
            # Penalize mismatches between header and actual line counts
            if plus_count != hunk['count2']:
                score -= 0.1
            if minus_count != hunk['count1']:
                score -= 0.1
                
            # Penalize very large hunks (they're more error-prone)
            if len(hunk['lines']) > 50:
                score -= 0.1
        
        # Normalize score to 0.0-1.0 range
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def extract_from_llm_response(response: str) -> List['UnifiedDiff']:
        """
        Extract unified diff blocks from an LLM response and return a list of UnifiedDiff objects.
        
        Handles both code-fenced blocks and direct diff content in the response.
        Recognizes hunk headers even without code fences and extracts the
        associated content.
        
        Args:
            response: The full response from an LLM
            
        Returns:
            A list of UnifiedDiff objects
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
        
        # Create a list of UnifiedDiff objects
        result = []
        for block in diff_blocks:
            diff = cls.from_string(block)
            if diff.hunks:  # Only add non-empty diffs
                result.append(diff)
                
        return result
    
    @classmethod
    def generate_diff(cls, before_code: str, after_code: str) -> 'UnifiedDiff':
        """
        Generate a UnifiedDiff object representing the changes between before and after code versions.
        
        Args:
            before_code: The original code snippet
            after_code: The fixed/modified code snippet
            
        Returns:
            A UnifiedDiff object representing the changes
        """
        # Split code into lines
        before_lines = before_code.splitlines()
        after_lines = after_code.splitlines()
        
        # Default context lines
        context_lines = 3
        
        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            before_lines, 
            after_lines,
            n=context_lines,
            lineterm=''
        ))
        
        # Skip the file headers (first two lines)
        if len(diff_lines) >= 2:
            diff_lines = diff_lines[2:]
        
        # Parse the diff into hunks
        hunks = []
        current_hunk = None
        
        for line in diff_lines:
            # Look for hunk headers
            hunk_match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)', line)
            
            if hunk_match:
                # If we were processing a hunk, add it to the list
                if current_hunk is not None:
                    hunks.append(current_hunk)
                
                # Parse the hunk header
                start1 = int(hunk_match.group(1))
                count1 = int(hunk_match.group(2) or 1)
                start2 = int(hunk_match.group(3))
                count2 = int(hunk_match.group(4) or 1)
                heading = hunk_match.group(5).strip()
                
                # Create a new hunk
                current_hunk = {
                    'start1': start1,
                    'count1': count1,
                    'start2': start2,
                    'count2': count2,
                    'heading': heading,
                    'lines': []
                }
            elif current_hunk is not None:
                # Add the line to the current hunk
                current_hunk['lines'].append(line)
        
        # Add the last hunk if there is one
        if current_hunk is not None:
            hunks.append(current_hunk)
            
        return cls(hunks, context_lines)
    
    def to_string(self) -> str:
        """
        Convert this diff object to its string representation.
        
        Returns:
            A string containing the unified diff
        """
        lines = []
        
        # Add file headers (placeholder)
        lines.append('--- a/file')
        lines.append('+++ b/file')
        
        # Add each hunk
        for hunk in self.hunks:
            # Add the hunk header
            header = f"@@ -{hunk['start1']},{hunk['count1']} +{hunk['start2']},{hunk['count2']} @@"
            if hunk.get('heading'):
                header += f" {hunk['heading']}"
            lines.append(header)
            
            # Add the hunk lines
            lines.extend(hunk['lines'])
            
        # Join the lines with newlines
        return '\n'.join(lines)


if __name__ == "__main__":
    """Example script demonstrating the use of different diff implementations."""
    
    print("=== Diff Utility Example ===\n")
    
    # Sample code
    before_code = """def calculate(x, y):
        # Add two numbers
        result = x + y
        return result"""
    
    after_code = """def calculate(x, y):
        # Add two numbers and multiply by 2
        result = (x + y) * 2
        return result"""
    
    print("Before code:")
    print("------------")
    print(before_code)
    print("\nAfter code:")
    print("-----------")
    print(after_code)
    
    # Get different diff implementations
    search_replace_diff = get_diff('search_replace')
    unified_diff = get_diff('unified')
    
    # Generate diffs
    sr_diff = search_replace_diff.generate_diff(before_code, after_code)
    unified = unified_diff.generate_diff(before_code, after_code)
    
    # Display diffs
    print("\nSearch/Replace Diff:")
    print("-------------------")
    print(sr_diff)
    
    print("\nUnified Diff:")
    print("-------------")
    print(unified)
    
    # Apply diffs
    sr_result = search_replace_diff.apply_diff(before_code, sr_diff)
    unified_result = unified_diff.apply_diff(before_code, unified)
    
    # Verify results
    print("\nVerification:")
    print("-------------")
    print(f"Search/Replace result matches: {sr_result == after_code}")
    print(f"Unified Diff result matches: {unified_result == after_code}")
    
    # Custom unified diff with more context lines
    custom_unified = get_diff('unified', context_lines=5)
    custom_diff = custom_unified.generate_diff(before_code, after_code)
    
    print("\nUnified Diff with 5 context lines:")
    print("---------------------------------")
    print(custom_diff)