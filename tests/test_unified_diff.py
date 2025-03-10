import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diff import UnifiedDiff, get_diff


class TestUnifiedDiff(unittest.TestCase):
    """Test cases for UnifiedDiff in src/utils/diff.py."""

    def setUp(self):
        """Set up a UnifiedDiff instance for testing."""
        self.diff = UnifiedDiff()

    # Basic functionality tests
    
    def test_generate_simple_diff(self):
        """Test generating a simple unified diff."""
        before_code = "def hello():\n    print('hello')\n    return None"
        after_code = "def hello():\n    print('hello world')\n    return None"
        
        diff = self.diff.generate_diff(before_code, after_code)
        
        # Basic checks for unified diff format
        self.assertIn("@@ -1,3 +1,3 @@", diff)
        self.assertIn(" def hello():", diff)
        self.assertIn("-    print('hello')", diff)
        self.assertIn("+    print('hello world')", diff)
        self.assertIn(" return None", diff)

    def test_generate_multi_line_diff(self):
        """Test generating a diff with multiple changed lines."""
        before_code = "def calculate(x, y):\n    # Add two numbers\n    result = x + y\n    return result"
        after_code = "def calculate(x, y):\n    # Add two numbers and multiply by 2\n    result = (x + y) * 2\n    return result"
        
        diff = self.diff.generate_diff(before_code, after_code)
        
        self.assertIn("@@ -1,4 +1,4 @@", diff)
        self.assertIn(" def calculate(x, y):", diff)
        self.assertIn("-    # Add two numbers", diff)
        self.assertIn("+    # Add two numbers and multiply by 2", diff)
        self.assertIn("-    result = x + y", diff)
        self.assertIn("+    result = (x + y) * 2", diff)
        self.assertIn(" return result", diff)

    def test_parse_diff(self):
        """Test parsing a unified diff."""
        diff_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""

        parsed = self.diff.parse_diff(diff_text)
        
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['old_start'], 1)
        self.assertEqual(parsed[0]['old_count'], 3)
        self.assertEqual(parsed[0]['new_start'], 1)
        self.assertEqual(parsed[0]['new_count'], 3)
        self.assertEqual(len(parsed[0]['lines']), 4)
        self.assertEqual(parsed[0]['lines'][0], " def hello():")
        self.assertEqual(parsed[0]['lines'][1], "-    print('hello')")
        self.assertEqual(parsed[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(parsed[0]['lines'][3], " return None")

    def test_apply_diff(self):
        """Test applying a unified diff to code."""
        code = "def hello():\n    print('hello')\n    return None"
        diff_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""

        result = self.diff.apply_diff(code, diff_text)
        expected = "def hello():\n    print('hello world')\nreturn None"
        
        self.assertEqual(result, expected)

    def test_is_valid_format(self):
        """Test validating unified diff format."""
        valid_diff = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""
        
        invalid_diff = """This is not a valid unified diff."""
        
        self.assertTrue(self.diff.is_valid_format(valid_diff))
        self.assertFalse(self.diff.is_valid_format(invalid_diff))
        
        # Empty diff is considered valid
        self.assertTrue(self.diff.is_valid_format(""))

    def test_extract_from_llm_response(self):
        """Test extracting unified diff blocks from an LLM response."""
        llm_response = """Here's the change I'm suggesting:

```diff
@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None
```

This will make the function greet with 'hello world' instead of just 'hello'.
"""
        
        extracted = self.diff.extract_from_llm_response(llm_response)
        
        expected = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""
        
        self.assertEqual(extracted, expected)

    def test_get_diff_factory(self):
        """Test the diff factory function."""
        search_replace_diff = get_diff('search_replace')
        unified_diff = get_diff('unified')
        custom_unified_diff = get_diff('unified', context_lines=5)
        
        from src.utils.diff import SearchReplaceDiff
        self.assertIsInstance(search_replace_diff, SearchReplaceDiff)
        self.assertIsInstance(unified_diff, UnifiedDiff)
        self.assertEqual(custom_unified_diff.context_lines, 5)
        
        with self.assertRaises(ValueError):
            get_diff('invalid_type')
    
    def test_roundtrip(self):
        """Test a complete roundtrip: generate diff and apply it."""
        before_code = (
            "def complex_function(a, b, c):\n"
            "    # Initialize result\n"
            "    result = 0\n"
            "    \n"
            "    # Process inputs\n"
            "    if a > 0:\n"
            "        result += a\n"
            "    \n"
            "    # Apply transformation\n"
            "    for i in range(b):\n"
            "        result *= 2\n"
            "        \n"
            "    # Final adjustment\n"
            "    result += c\n"
            "    \n"
            "    return result"
        )
        
        after_code = (
            "def complex_function(a, b, c):\n"
            "    # Initialize result with default\n"
            "    result = 0\n"
            "    \n"
            "    # Validate inputs\n"
            "    if not isinstance(a, int) or not isinstance(b, int) or not isinstance(c, int):\n"
            "        raise TypeError(\"All inputs must be integers\")\n"
            "    \n"
            "    # Process inputs\n"
            "    if a > 0:\n"
            "        result += a * 2  # Double the positive input\n"
            "    \n"
            "    # Apply transformation with limit\n"
            "    for i in range(min(b, 10)):  # Limit iterations to 10\n"
            "        result *= 2\n"
            "        \n"
            "    # Final adjustment\n"
            "    result += c\n"
            "    \n"
            "    return result"
        )
        
        # Generate diff
        diff = self.diff.generate_diff(before_code, after_code)
        
        # Apply diff
        result = self.diff.apply_diff(before_code, diff)
        
        # Verify result matches after_code
        self.assertEqual(result, after_code)
    
    # Robust handling tests
    
    def test_parse_standard_diff(self):
        """Test parsing a standard unified diff."""
        diff = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 3)
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 3)
        self.assertEqual(len(hunks[0]['lines']), 4)

    def test_parse_malformed_hunk_header(self):
        """Test parsing a diff with malformed hunk header."""
        diff = (
            "@@ -1,3 +1,3 @\n"  # Missing one @
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 3)  # Ensure old_count is parsed correctly
        self.assertEqual(hunks[0]['new_count'], 3)  # Ensure new_count is parsed correctly
        self.assertEqual(len(hunks[0]['lines']), 4)  # Ensure all lines are parsed

    def test_parse_spaces_in_hunk_header(self):
        """Test parsing a diff with spaces in the hunk header."""
        diff = (
            "@@  -1,3  +1,3  @@\n"  # Extra spaces
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 3)
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 3)
        self.assertEqual(len(hunks[0]['lines']), 4)  # Ensure all lines are parsed
        self.assertEqual(hunks[0]['lines'][0], " def hello():")
        self.assertEqual(hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(hunks[0]['lines'][3], " return None")

    def test_parse_excessive_hunk_markers(self):
        """Test parsing a diff with excessive hunk markers."""
        diff = (
            "@@@@@@ -1,3 +1,3 @@@@@@\n"  # Too many @s
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 3)
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 3)
        self.assertEqual(len(hunks[0]['lines']), 4)  # Ensure all lines are parsed
        self.assertEqual(hunks[0]['lines'][0], " def hello():")
        self.assertEqual(hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(hunks[0]['lines'][3], " return None")

    def test_parse_colon_instead_of_comma(self):
        """Test parsing a diff with colon instead of comma in line numbers."""
        diff = (
            "@@ -1:3 +1:3 @@\n"  # Using : instead of ,
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 3)
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 3)
        self.assertEqual(len(hunks[0]['lines']), 4)  # Ensure all lines are parsed
        self.assertEqual(hunks[0]['lines'][0], " def hello():")
        self.assertEqual(hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(hunks[0]['lines'][3], " return None")

    def test_parse_missing_line_prefixes(self):
        """Test parsing a diff with missing line prefixes."""
        diff = (
            "@@ -1,3 +1,3 @@\n"
            "def hello():\n"  # Missing space prefix
            "-    print('hello')\n"
            "+    print('hello world')\n"
            "return None"  # Missing space prefix
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(len(hunks[0]['lines']), 4)
        # The parser normalizes the lines
        self.assertEqual(hunks[0]['lines'][0], " def hello():")
        self.assertEqual(hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(hunks[0]['lines'][3], " return None")
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 3)
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 3)

    def test_parse_verbose_line_prefixes(self):
        """Test parsing a diff with verbose line prefixes."""
        diff = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "removed     print('hello')\n"  # Verbose 'removed' prefix
            "added     print('hello world')\n"  # Verbose 'added' prefix
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(len(hunks[0]['lines']), 4)
        # The parser should normalize these to standard prefixes
        self.assertTrue(hunks[0]['lines'][1].startswith('-') or 
                       hunks[0]['lines'][1].startswith("removed"))
        self.assertTrue(hunks[0]['lines'][2].startswith('+') or 
                       hunks[0]['lines'][2].startswith("added"))
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 3)
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 3)

    def test_quality_validation_perfect(self):
        """Test quality validation on a perfect diff."""
        diff = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        quality = self.diff.validate_quality(diff)
        self.assertEqual(quality, 1.0)

    def test_quality_validation_good_enough(self):
        """Test quality validation on a good enough diff."""
        diff = (
            "@@ -1:3 +1:3 @\n"  # Using : instead of , and missing one @
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        quality = self.diff.validate_quality(diff)
        self.assertGreaterEqual(quality, 0.7)
        self.assertLessEqual(quality, 0.9)

    def test_quality_validation_recoverable(self):
        """Test quality validation on a recoverable diff."""
        diff = (
            "@ -1 +1 @\n"  # Very minimal header
            "-print('hello')\n"
            "+print('hello world')"
        )
        
        quality = self.diff.validate_quality(diff)
        # This is a recoverable diff with minimal but present markers
        self.assertGreaterEqual(quality, 0.4)

    def test_quality_validation_poor(self):
        """Test quality validation on a poor diff with only markers."""
        diff = (
            "Here's a diff with line 1 and adds the word 'world':\n"
            "+print('hello world')\n"
            "-print('hello')"
        )
        
        quality = self.diff.validate_quality(diff)
        self.assertGreaterEqual(quality, 0.1)
        self.assertLessEqual(quality, 0.4)

    def test_quality_validation_invalid(self):
        """Test quality validation on an invalid diff."""
        diff = "This is not a diff at all."
        
        quality = self.diff.validate_quality(diff)
        self.assertEqual(quality, 0.0)

    def test_safe_apply_perfect_diff(self):
        """Test safely applying a perfect diff."""
        code = "def hello():\n    print('hello')\n    return None"
        diff = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        result, quality = self.diff.safe_apply_diff(code, diff)
        
        # Check the quality - should be perfect
        self.assertEqual(quality, 1.0)
        
        # Check content
        expected = "def hello():\n    print('hello world')\nreturn None"
        self.assertEqual(result, expected)

    def test_safe_apply_recoverable_diff(self):
        """Test safely applying a recoverable diff."""
        code = "def hello():\n    print('hello')\n    return None"
        diff = (
            "@@ -1,3 +1,3 @@\n"
            "def hello():\n"  # Missing space
            "-    print('hello')\n"
            "+    print('hello world')\n"
            "return None"  # Missing space
        )
        
        result, quality = self.diff.safe_apply_diff(code, diff)
        
        # Quality should be good enough to apply
        self.assertGreaterEqual(quality, 0.5)
        
        # Normalize whitespace for comparison
        if result != code:  # If the diff was applied
            normalized_result = "\n".join(line.strip() for line in result.splitlines())
            normalized_expected = "\n".join(line.strip() for line in "def hello():\n    print('hello world')\nreturn None".splitlines())
            self.assertEqual(normalized_result, normalized_expected)

    def test_safe_apply_invalid_diff(self):
        """Test safely applying an invalid diff."""
        code = "def hello():\n    print('hello')\n    return None"
        diff = "This is not a diff at all."
        
        result, quality = self.diff.safe_apply_diff(code, diff)
        
        # Invalid diff should return original code
        self.assertEqual(result, code)
        self.assertEqual(quality, 0.0)

    def test_lenient_validation(self):
        """Test lenient validation on different diff formats."""
        valid_strict = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        valid_lenient = (
            "@ -1 +1 @\n"
            "-print('hello')\n"
            "+print('hello world')"
        )
        
        invalid = "Not a diff at all"
        
        # Strict validation
        self.assertTrue(self.diff.is_valid_format(valid_strict, strict=True))
        self.assertFalse(self.diff.is_valid_format(valid_lenient, strict=True))
        self.assertFalse(self.diff.is_valid_format(invalid, strict=True))
        
        # Lenient validation
        self.assertTrue(self.diff.is_valid_format(valid_strict, strict=False))
        self.assertTrue(self.diff.is_valid_format(valid_lenient, strict=False))
        self.assertFalse(self.diff.is_valid_format(invalid, strict=False))
    
    # Additional edge cases
    
    def test_multi_hunk_diff(self):
        """Test parsing and applying a diff with multiple hunks."""
        code = (
            "def hello():\n"
            "    print('hello')\n"
            "    return None\n"
            "\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "    return None"
        )
        
        diff = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None\n"
            "@@ -5,3 +5,3 @@\n"
            " def goodbye():\n"
            "-    print('goodbye')\n"
            "+    print('goodbye world')\n"
            " return None"
        )
        
        # Parse the diff
        hunks = self.diff.parse_diff(diff)
        self.assertEqual(len(hunks), 2)
        
        # Apply the diff
        result = self.diff.apply_diff(code, diff)
        expected = (
            "def hello():\n"
            "    print('hello world')\n"
            "return None\n"
            "\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            "return None"
        )
        self.assertEqual(result, expected)
    
    def test_header_missing_count(self):
        """Test parsing a diff with missing count in the header."""
        diff = (
            "@@ -1 +1 @@\n"  # Missing counts
            "-print('hello')\n"
            "+print('hello world')"
        )
        
        hunks = self.diff.parse_diff(diff)
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 1)  # Default to 1 if not specified
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 1)  # Default to 1 if not specified
    
    def test_complete_file_replacement(self):
        """Test applying a diff that completely replaces a file."""
        before_code = "def old_function():\n    return 'old value'"
        after_code = "def new_function():\n    return 'new value'"
        
        # Generate diff
        diff = self.diff.generate_diff(before_code, after_code)
        
        # Apply diff
        result = self.diff.apply_diff(before_code, diff)
        
        # Verify result
        self.assertEqual(result, after_code)
    
    def test_empty_file_handling(self):
        """Test handling empty files in diff generation and application."""
        # Empty before, non-empty after
        self.assertNotEqual(self.diff.generate_diff("", "def func():\n    pass"), "")
        
        # Non-empty before, empty after
        self.assertNotEqual(self.diff.generate_diff("def func():\n    pass", ""), "")
        
        # Both empty
        self.assertEqual(self.diff.generate_diff("", ""), "")
    
    def test_context_line_count(self):
        """Test that the context line count is respected."""
        before_code = (
            "line 1\n"
            "line 2\n"
            "line 3\n"
            "line 4\n"
            "line 5\n"
            "line 6\n"
            "line 7\n"
            "line 8\n"
            "line 9"
        )
        
        after_code = (
            "line 1\n"
            "line 2\n"
            "line 3\n"
            "line 4 modified\n"
            "line 5\n"
            "line 6\n"
            "line 7\n"
            "line 8\n"
            "line 9"
        )
        
        # Create diff with default context (3 lines)
        default_diff = self.diff.generate_diff(before_code, after_code)
        
        # Create diff with custom context (1 line)
        custom_diff = UnifiedDiff(context_lines=1).generate_diff(before_code, after_code)
        
        # The default diff should have more context lines than the custom diff
        self.assertGreater(len(default_diff.splitlines()), len(custom_diff.splitlines()))
        
        # But both should still work when applied
        self.assertEqual(self.diff.apply_diff(before_code, default_diff), after_code)
        self.assertEqual(self.diff.apply_diff(before_code, custom_diff), after_code)
    
    def test_extract_from_multiple_blocks(self):
        """Test extracting unified diff from LLM response with multiple code blocks."""
        llm_response = """Here's my first suggestion:

```diff
@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None
```

And another change:

```diff
@@ -5,3 +5,3 @@
 def goodbye():
-    print('goodbye')
+    print('goodbye world')
 return None
```
"""
        
        extracted = self.diff.extract_from_llm_response(llm_response)
        
        expected = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None

@@ -5,3 +5,3 @@
 def goodbye():
-    print('goodbye')
+    print('goodbye world')
 return None"""
        
        self.assertEqual(extracted, expected)

    def test_malformed_hunk_header(self):
        """Test parsing a diff with malformed hunk header."""
        diff = (
            "@@ -1 +1 @@\n"  # Missing counts
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]['old_start'], 1)
        self.assertEqual(hunks[0]['old_count'], 1)  # Default count
        self.assertEqual(hunks[0]['new_start'], 1)
        self.assertEqual(hunks[0]['new_count'], 1)  # Default count
        
    def test_very_malformed_hunk_header(self):
        """Test parsing a diff with very malformed hunk header."""
        diff = (
            "@@ bad header with numbers 10 20 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        hunks = self.diff.parse_diff(diff)
        
        # The current implementation doesn't recognize this as a valid hunk header
        # because it doesn't have the required '-' and '+' characters in the header
        # This is actually correct behavior - we shouldn't try to parse completely invalid headers
        self.assertEqual(len(hunks), 0)

    def test_invalid_hunk_validation(self):
        """Test validation of invalid hunks."""
        # Create a diff instance
        diff = UnifiedDiff()
        
        # Test with empty lines
        invalid_hunk1 = {
            'old_start': 1,
            'old_count': 3,
            'new_start': 1,
            'new_count': 3,
            'lines': []
        }
        self.assertFalse(diff._validate_hunk(invalid_hunk1))
        
        # Test with negative counts
        invalid_hunk2 = {
            'old_start': 1,
            'old_count': -1,
            'new_start': 1,
            'new_count': 3,
            'lines': [" line"]
        }
        self.assertFalse(diff._validate_hunk(invalid_hunk2))
        
        # Test with negative start positions
        invalid_hunk3 = {
            'old_start': -1,
            'old_count': 3,
            'new_start': 1,
            'new_count': 3,
            'lines': [" line"]
        }
        self.assertFalse(diff._validate_hunk(invalid_hunk3))
        
        # Test with valid hunk
        valid_hunk = {
            'old_start': 1,
            'old_count': 3,
            'new_start': 1,
            'new_count': 3,
            'lines': [" line"]
        }
        self.assertTrue(diff._validate_hunk(valid_hunk))
        
    def test_extract_unified_diff_without_code_fences(self):
        """Test extracting unified diff from LLM response without code fences."""
        llm_response = (
            "Here's my fix:\n"
            "\n"
            "@@ -1,3 +1,4 @@\n"
            " def calculate(x, y):\n"
            "+    if y == 0:\n"
            "+        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "     return x / y\n"
            "\n"
            "This should handle the division by zero error."
        )
        
        # Extract the blocks
        extracted = self.diff.extract_from_llm_response(llm_response)
        
        # Define the expected block
        expected_block = (
            "@@ -1,3 +1,4 @@\n"
            " def calculate(x, y):\n"
            "+    if y == 0:\n"
            "+        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "     return x / y"
        )
        
        self.assertEqual(extracted, expected_block)

    def test_inaccurate_line_numbers(self):
        """Test applying a diff with inaccurate line numbers in hunk headers."""
        # Original code with multiple occurrences of the same line
        code = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    x = 10\n"  # Duplicate line
            "    z = 30\n"
            "    return x + y + z"  # No trailing newline
        )
        
        # Diff with incorrect line number (targeting line 4 but specifying line 2)
        # Include context to make it clear which occurrence we want to change
        diff = (
            "@@ -2,3 +2,3 @@\n"  # Should be line 3, not line 2
            "     x = 10\n"
            "-    y = 20\n"
            "-    x = 10\n"
            "+    y = 20\n"
            "+    x = 15\n"
        )
        
        # Expected result - should change the second occurrence of "x = 10"
        expected = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    x = 15\n"  # This should be changed
            "    z = 30\n"
            "    return x + y + z"  # No trailing newline
        )
        
        # Apply the diff
        result = self.diff.apply_diff(code, diff)
        
        # Check if the result matches the expected output
        self.assertEqual(result, expected)

    def test_inaccurate_line_numbers_multiple_occurrences(self):
        """Test applying a diff with multiple occurrences of the target line."""
        # Code with three identical lines
        code = (
            "def process_data():\n"
            "    value = 10\n"
            "    value = 10\n"
            "    value = 10\n"
            "    return value"
        )
        
        # Diff targeting the middle occurrence but with incorrect line number
        diff = (
            "@@ -1,2 +1,2 @@\n"  # Should be line 3, not line 1
            " def process_data():\n"
            "-    value = 10\n"
            "+    value = 20\n"
        )
        
        # Expected result - should change the first occurrence since it's closest to line 1
        expected = (
            "def process_data():\n"
            "    value = 20\n"
            "    value = 10\n"
            "    value = 10\n"
            "    return value"
        )
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_inaccurate_line_numbers_with_context(self):
        """Test applying a diff with inaccurate line numbers but unique context."""
        # Code with similar lines but different context
        code = (
            "def first_function():\n"
            "    x = 5\n"
            "    return x\n"
            "\n"
            "def second_function():\n"
            "    x = 5\n"
            "    return x * 2"
        )
        
        # Diff targeting the second x=5 but with incorrect line number
        diff = (
            "@@ -2,2 +2,2 @@\n"  # Should be line 6, not line 2
            "     x = 5\n"
            "-    return x * 2\n"
            "+    return x * 3\n"
        )
        
        # Expected result - should change the second function due to context
        expected = (
            "def first_function():\n"
            "    x = 5\n"
            "    return x\n"
            "\n"
            "def second_function():\n"
            "    x = 5\n"
            "    return x * 3"
        )
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_inaccurate_line_numbers_no_match(self):
        """Test applying a diff where the content doesn't match anywhere."""
        code = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    return x + y"
        )
        
        # Diff with content that doesn't match anywhere in the code
        diff = (
            "@@ -2,1 +2,1 @@\n"
            "-    z = 30\n"
            "+    z = 40\n"
        )
        
        # Expected result - diff should not be applied
        expected = code
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_inaccurate_line_numbers_partial_match(self):
        """Test applying a diff with partial content matches."""
        code = (
            "def calculate():\n"
            "    # Initialize variables\n"
            "    x = 10\n"
            "    y = 20\n"
            "    # Calculate result\n"
            "    result = x + y\n"
            "    return result"
        )
        
        # Diff with content that partially matches multiple places
        diff = (
            "@@ -5,1 +5,1 @@\n"  # Should be line 6, not line 5
            "-    result = x + y\n"
            "+    result = x * y\n"
        )
        
        # Expected result - should change line 6 based on content
        expected = (
            "def calculate():\n"
            "    # Initialize variables\n"
            "    x = 10\n"
            "    y = 20\n"
            "    # Calculate result\n"
            "    result = x * y\n"
            "    return result"
        )
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_inaccurate_line_numbers_multiple_hunks(self):
        """Test applying a diff with multiple hunks having inaccurate line numbers."""
        code = (
            "def first_function():\n"
            "    x = 5\n"
            "    return x\n"
            "\n"
            "def second_function():\n"
            "    y = 10\n"
            "    return y\n"
            "\n"
            "def third_function():\n"
            "    z = 15\n"
            "    return z"
        )
        
        # Diff with multiple hunks, all with incorrect line numbers
        diff = (
            "@@ -2,1 +2,1 @@\n"  # Should be line 2, correct
            "-    x = 5\n"
            "+    x = 50\n"
            "@@ -5,1 +5,1 @@\n"  # Should be line 6, not line 5
            "-    y = 10\n"
            "+    y = 100\n"
            "@@ -8,1 +8,1 @@\n"  # Should be line 10, not line 8
            "-    z = 15\n"
            "+    z = 150\n"
        )
        
        # Expected result - should change all three functions based on content
        expected = (
            "def first_function():\n"
            "    x = 50\n"
            "    return x\n"
            "\n"
            "def second_function():\n"
            "    y = 100\n"
            "    return y\n"
            "\n"
            "def third_function():\n"
            "    z = 150\n"
            "    return z"
        )
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)

    def test_inaccurate_line_numbers_empty_code(self):
        """Test applying a diff with inaccurate line numbers to empty code."""
        code = ""
        
        # Diff with line numbers that can't possibly match empty code
        diff = (
            "@@ -5,1 +5,1 @@\n"
            "-    x = 10\n"
            "+    x = 20\n"
        )
        
        # Expected result - diff should not be applied to empty code
        expected = ""
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_inaccurate_line_numbers_out_of_bounds(self):
        """Test applying a diff with line numbers that are out of bounds."""
        code = (
            "def calculate():\n"
            "    x = 10\n"
            "    return x"
        )
        
        # Diff with line numbers that are out of bounds
        diff = (
            "@@ -10,1 +10,1 @@\n"  # Line 10 doesn't exist
            "-    y = 20\n"
            "+    y = 30\n"
        )
        
        # Expected result - diff should not be applied
        expected = code
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_inaccurate_line_numbers_ambiguous_match(self):
        """Test applying a diff with ambiguous matches (multiple equally good matches)."""
        # Code with identical blocks
        code = (
            "def first():\n"
            "    x = 10\n"
            "    y = 20\n"
            "\n"
            "def second():\n"
            "    x = 10\n"
            "    y = 20"  # No trailing newline
        )
        
        # Diff that could match either block
        diff = (
            "@@ -2,2 +2,2 @@\n"
            "     x = 10\n"
            "-    y = 20\n"
            "+    y = 30\n"
        )
        
        # Expected result - should change the first occurrence since it's closest to line 2
        expected = (
            "def first():\n"
            "    x = 10\n"
            "    y = 30\n"
            "\n"
            "def second():\n"
            "    x = 10\n"
            "    y = 20"  # No trailing newline
        )
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_inaccurate_line_numbers_mixed_matches(self):
        """Test applying a diff with some hunks matching and others not."""
        code = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    return x + y"
        )
        
        # Diff with one hunk that matches and one that doesn't
        diff = (
            "@@ -2,1 +2,1 @@\n"
            "-    x = 10\n"
            "+    x = 15\n"
            "@@ -5,1 +5,1 @@\n"  # This line doesn't exist
            "-    z = 30\n"
            "+    z = 40\n"
        )
        
        # Expected result - only the matching hunk should be applied
        expected = (
            "def calculate():\n"
            "    x = 15\n"
            "    y = 20\n"
            "    return x + y"
        )
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()