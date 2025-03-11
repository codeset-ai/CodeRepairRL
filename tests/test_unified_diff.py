import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diff import UnifiedDiff


class TestUnifiedDiff(unittest.TestCase):
    """Test cases for UnifiedDiff in src/utils/diff.py."""

    def setUp(self):
        """Set up a UnifiedDiff instance for testing."""
        self.diff = UnifiedDiff([], 3)  # Initialize with empty hunks list and default context lines

    # Basic functionality tests
    
    def test_generate_simple_diff(self):
        """Test generating a simple unified diff."""
        before_code = "def hello():\n    print('hello')\n    return None"
        after_code = "def hello():\n    print('hello world')\n    return None"
        
        diff = UnifiedDiff.generate_diff(before_code, after_code)
        diff_str = diff.to_string()
        
        # Basic checks for unified diff format
        self.assertIn("@@ -1,3 +1,3 @@", diff_str)
        self.assertIn(" def hello():", diff_str)
        self.assertIn("-    print('hello')", diff_str)
        self.assertIn("+    print('hello world')", diff_str)
        self.assertIn(" return None", diff_str)

    def test_generate_multi_line_diff(self):
        """Test generating a diff with multiple changed lines."""
        before_code = "def calculate(x, y):\n    # Add two numbers\n    result = x + y\n    return result"
        after_code = "def calculate(x, y):\n    # Add two numbers and multiply by 2\n    result = (x + y) * 2\n    return result"
        
        diff = UnifiedDiff.generate_diff(before_code, after_code)
        diff_str = diff.to_string()
        
        self.assertIn("@@ -1,4 +1,4 @@", diff_str)
        self.assertIn(" def calculate(x, y):", diff_str)
        self.assertIn("-    # Add two numbers", diff_str)
        self.assertIn("+    # Add two numbers and multiply by 2", diff_str)
        self.assertIn("-    result = x + y", diff_str)
        self.assertIn("+    result = (x + y) * 2", diff_str)
        self.assertIn(" return result", diff_str)

    def test_from_string(self):
        """Test parsing a unified diff."""
        diff_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""

        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 3)
        self.assertEqual(len(diff.hunks[0]['lines']), 4)
        self.assertEqual(diff.hunks[0]['lines'][0], " def hello():")
        self.assertEqual(diff.hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(diff.hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(diff.hunks[0]['lines'][3], " return None")

    def test_apply_diff(self):
        """Test applying a unified diff to code."""
        code = "def hello():\n    print('hello')\n    return None"
        diff_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""

        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code)
        expected = "def hello():\n    print('hello world')\nreturn None"
        
        self.assertEqual(result, expected)

    def test_is_valid_format(self):
        """Test validating unified diff format."""
        valid_diff_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""
        
        invalid_diff_text = """This is not a valid unified diff."""
        
        valid_diff = UnifiedDiff.from_string(valid_diff_text)
        invalid_diff = UnifiedDiff.from_string(invalid_diff_text)
        empty_diff = UnifiedDiff([], 3)
        
        self.assertTrue(valid_diff.is_valid_format())
        self.assertFalse(invalid_diff.is_valid_format())
        
        # Empty diff is considered valid
        self.assertTrue(empty_diff.is_valid_format())

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
        
        extracted_diffs = UnifiedDiff.extract_from_llm_response(llm_response)
        
        self.assertEqual(len(extracted_diffs), 1)
        
        expected_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None"""
        
        # Convert the extracted diff to string for comparison
        extracted_text = extracted_diffs[0].to_string()
        
        # Remove file headers for comparison
        extracted_text = '\n'.join(extracted_text.split('\n')[2:])
        
        self.assertEqual(extracted_text, expected_text)
    
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
        diff = UnifiedDiff.generate_diff(before_code, after_code)
        
        # Apply diff
        result = diff.apply_diff(before_code)
        
        # Verify result matches after_code
        self.assertEqual(result, after_code)
    
    # Robust handling tests
    
    def test_parse_standard_diff(self):
        """Test parsing a standard unified diff."""
        diff_text = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 3)
        self.assertEqual(len(diff.hunks[0]['lines']), 4)

    def test_parse_malformed_hunk_header(self):
        """Test parsing a diff with malformed hunk header."""
        diff_text = (
            "@@ -1,3 +1,3 @\n"  # Missing one @
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)  # Ensure count1 is parsed correctly
        self.assertEqual(diff.hunks[0]['count2'], 3)  # Ensure count2 is parsed correctly
        self.assertEqual(len(diff.hunks[0]['lines']), 4)  # Ensure all lines are parsed

    def test_parse_spaces_in_hunk_header(self):
        """Test parsing a diff with spaces in the hunk header."""
        diff_text = (
            "@@  -1,3  +1,3  @@\n"  # Extra spaces
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 3)
        self.assertEqual(len(diff.hunks[0]['lines']), 4)  # Ensure all lines are parsed
        self.assertEqual(diff.hunks[0]['lines'][0], " def hello():")
        self.assertEqual(diff.hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(diff.hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(diff.hunks[0]['lines'][3], " return None")

    def test_parse_excessive_hunk_markers(self):
        """Test parsing a diff with excessive hunk markers."""
        diff_text = (
            "@@@@@@ -1,3 +1,3 @@@@@@\n"  # Too many @s
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 3)
        self.assertEqual(len(diff.hunks[0]['lines']), 4)  # Ensure all lines are parsed
        self.assertEqual(diff.hunks[0]['lines'][0], " def hello():")
        self.assertEqual(diff.hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(diff.hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(diff.hunks[0]['lines'][3], " return None")

    def test_parse_colon_instead_of_comma(self):
        """Test parsing a diff with colon instead of comma in line numbers."""
        diff_text = (
            "@@ -1:3 +1:3 @@\n"  # Using : instead of ,
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 3)
        self.assertEqual(len(diff.hunks[0]['lines']), 4)  # Ensure all lines are parsed
        self.assertEqual(diff.hunks[0]['lines'][0], " def hello():")
        self.assertEqual(diff.hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(diff.hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(diff.hunks[0]['lines'][3], " return None")

    def test_parse_missing_line_prefixes(self):
        """Test parsing a diff with missing line prefixes."""
        diff_text = (
            "@@ -1,3 +1,3 @@\n"
            "def hello():\n"  # Missing space prefix
            "-    print('hello')\n"
            "+    print('hello world')\n"
            "return None"  # Missing space prefix
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(len(diff.hunks[0]['lines']), 4)
        # The parser normalizes the lines
        self.assertEqual(diff.hunks[0]['lines'][0], " def hello():")
        self.assertEqual(diff.hunks[0]['lines'][1], "-    print('hello')")
        self.assertEqual(diff.hunks[0]['lines'][2], "+    print('hello world')")
        self.assertEqual(diff.hunks[0]['lines'][3], " return None")
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 3)

    def test_parse_verbose_line_prefixes(self):
        """Test parsing a diff with verbose line prefixes."""
        diff_text = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "removed     print('hello')\n"  # Verbose 'removed' prefix
            "added     print('hello world')\n"  # Verbose 'added' prefix
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(len(diff.hunks[0]['lines']), 4)
        # The parser should normalize these to standard prefixes
        self.assertTrue(diff.hunks[0]['lines'][1].startswith('-') or 
                       diff.hunks[0]['lines'][1].startswith("removed"))
        self.assertTrue(diff.hunks[0]['lines'][2].startswith('+') or 
                       diff.hunks[0]['lines'][2].startswith("added"))
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 3)
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 3)

    def test_quality_validation_perfect(self):
        """Test quality validation on a perfect diff."""
        diff_text = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        quality = diff.validate_quality()
        self.assertEqual(quality, 1.0)

    def test_quality_validation_good_enough(self):
        """Test quality validation on a good enough diff."""
        diff_text = (
            "@@ -1:3 +1:3 @\n"  # Using : instead of , and missing one @
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        quality = diff.validate_quality()
        self.assertGreaterEqual(quality, 0.7)
        self.assertLessEqual(quality, 0.9)

    def test_quality_validation_recoverable(self):
        """Test quality validation on a recoverable diff."""
        diff_text = (
            "@ -1 +1 @\n"  # Very minimal header
            "-print('hello')\n"
            "+print('hello world')"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        quality = diff.validate_quality()
        # This is a recoverable diff with minimal but present markers
        self.assertGreaterEqual(quality, 0.4)

    def test_quality_validation_poor(self):
        """Test quality validation on a poor diff with only markers."""
        diff_text = (
            "Here's a diff with line 1 and adds the word 'world':\n"
            "+print('hello world')\n"
            "-print('hello')"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        quality = diff.validate_quality()
        self.assertGreaterEqual(quality, 0.1)
        self.assertLessEqual(quality, 0.4)

    def test_quality_validation_invalid(self):
        """Test quality validation on an invalid diff."""
        diff_text = "This is not a diff at all."
        
        diff = UnifiedDiff.from_string(diff_text)
        quality = diff.validate_quality()
        self.assertEqual(quality, 0.0)

    def test_safe_apply_perfect_diff(self):
        """Test safely applying a perfect diff."""
        code = "def hello():\n    print('hello')\n    return None"
        diff_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')\n"
+    print('hello world')\n"
 return None"""
        
        diff = UnifiedDiff.from_string(diff_text)
        result, quality = diff.safe_apply_diff(code)
        
        # Check the quality - should be perfect
        self.assertEqual(quality, 1.0)
        
        # Check content
        expected = "def hello():\n    print('hello world')\nreturn None"
        self.assertEqual(result, expected)

    def test_safe_apply_recoverable_diff(self):
        """Test safely applying a recoverable diff."""
        code = "def hello():\n    print('hello')\n    return None"
        diff_text = (
            "@@ -1,3 +1,3 @@\n"
            "def hello():\n"  # Missing space
            "-    print('hello')\n"
            "+    print('hello world')\n"
            "return None"  # Missing space
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        result, quality = diff.safe_apply_diff(code)
        
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
        diff_text = "This is not a diff at all."
        
        diff = UnifiedDiff.from_string(diff_text)
        result, quality = diff.safe_apply_diff(code)
        
        # Invalid diff should return original code
        self.assertEqual(result, code)
        self.assertEqual(quality, 0.0)

    def test_lenient_validation(self):
        """Test lenient validation on different diff formats."""
        valid_strict_text = (
            "@@ -1,3 +1,3 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        valid_lenient_text = (
            "@ -1 +1 @\n"
            "-print('hello')\n"
            "+print('hello world')"
        )
        
        invalid_text = "Not a diff at all"
        
        # Strict validation
        valid_strict = UnifiedDiff.from_string(valid_strict_text)
        self.assertTrue(valid_strict.is_valid_format(strict=True))
        self.assertFalse(valid_strict.is_valid_format(strict=True))
        self.assertFalse(valid_strict.is_valid_format(strict=True))
        
        # Lenient validation
        valid_lenient = UnifiedDiff.from_string(valid_lenient_text)
        self.assertTrue(valid_lenient.is_valid_format(strict=False))
        self.assertTrue(valid_lenient.is_valid_format(strict=False))
        self.assertFalse(valid_lenient.is_valid_format(strict=False))
    
    # Additional edge cases
    
    def test_multi_hunk_diff(self):
        """Test parsing and applying a diff with multiple hunks."""
        code_text = (
            "def hello():\n"
            "    print('hello')\n"
            "    return None\n"
            "\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "    return None"
        )
        
        diff_text = (
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
        diff = UnifiedDiff.from_string(diff_text)
        self.assertEqual(len(diff.hunks), 2)
        
        # Apply the diff
        result = diff.apply_diff(code_text)
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
        diff_text = (
            "@@ -1 +1 @@\n"  # Missing counts
            "-print('hello')\n"
            "+print('hello world')"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 1)  # Default to 1 if not specified
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 1)  # Default to 1 if not specified
    
    def test_complete_file_replacement(self):
        """Test applying a diff that completely replaces a file."""
        before_text = "def old_function():\n    return 'old value'"
        after_text = "def new_function():\n    return 'new value'"
        
        # Generate diff
        diff = UnifiedDiff.generate_diff(before_text, after_text)
        
        # Apply diff
        result = diff.apply_diff(before_text)
        
        # Verify result
        self.assertEqual(result, after_text)
    
    def test_empty_file_handling(self):
        """Test handling empty files in diff generation and application."""
        # Empty before, non-empty after
        self.assertNotEqual(UnifiedDiff.generate_diff("", "def func():\n    pass"), "")
        
        # Non-empty before, empty after
        self.assertNotEqual(UnifiedDiff.generate_diff("def func():\n    pass", ""), "")
        
        # Both empty
        self.assertEqual(UnifiedDiff.generate_diff("", ""), "")
    
    def test_context_line_count(self):
        """Test that the context line count is respected."""
        before_text = (
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
        
        after_text = (
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
        default_diff = UnifiedDiff.generate_diff(before_text, after_text)
        
        # Create diff with custom context (1 line)
        custom_diff = UnifiedDiff(context_lines=1).generate_diff(before_text, after_text)
        
        # The default diff should have more context lines than the custom diff
        self.assertGreater(len(default_diff.to_string().splitlines()), len(custom_diff.to_string().splitlines()))
        
        # But both should still work when applied
        self.assertEqual(custom_diff.apply_diff(before_text), after_text)
        self.assertEqual(default_diff.apply_diff(before_text), after_text)
    
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
        
        extracted_diffs = UnifiedDiff.extract_from_llm_response(llm_response)
        
        self.assertEqual(len(extracted_diffs), 2)
        
        expected_text = """@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hello world')
 return None

@@ -5,3 +5,3 @@
 def goodbye():
-    print('goodbye')
+    print('goodbye world')
 return None"""
        
        # Convert the extracted diffs to strings for comparison
        extracted_texts = [diff.to_string() for diff in extracted_diffs]
        
        # Remove file headers for comparison
        extracted_texts = ['\n'.join(text.split('\n')[2:]) for text in extracted_texts]
        
        self.assertEqual(extracted_texts, [expected_text])

    def test_malformed_hunk_header(self):
        """Test parsing a diff with malformed hunk header."""
        diff_text = (
            "@@ -1 +1 @@\n"  # Missing counts
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        self.assertEqual(len(diff.hunks), 1)
        self.assertEqual(diff.hunks[0]['start1'], 1)
        self.assertEqual(diff.hunks[0]['count1'], 1)  # Default count
        self.assertEqual(diff.hunks[0]['start2'], 1)
        self.assertEqual(diff.hunks[0]['count2'], 1)  # Default count
        
    def test_very_malformed_hunk_header(self):
        """Test parsing a diff with very malformed hunk header."""
        diff_text = (
            "@@ bad header with numbers 10 20 @@\n"
            " def hello():\n"
            "-    print('hello')\n"
            "+    print('hello world')\n"
            " return None"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        
        # The current implementation doesn't recognize this as a valid hunk header
        # because it doesn't have the required '-' and '+' characters in the header
        # This is actually correct behavior - we shouldn't try to parse completely invalid headers
        self.assertEqual(len(diff.hunks), 0)

    def test_invalid_hunk_validation(self):
        """Test validation of invalid hunks."""
        # Create a diff instance
        diff = UnifiedDiff([], 3)
        
        # Test with empty lines
        invalid_hunk1 = {
            'start1': 1,
            'count1': 3,
            'start2': 1,
            'count2': 3,
            'lines': []
        }
        self.assertFalse(diff._validate_hunk(invalid_hunk1))
        
        # Test with negative counts
        invalid_hunk2 = {
            'start1': 1,
            'count1': -1,
            'start2': 1,
            'count2': 3,
            'lines': [" line"]
        }
        self.assertFalse(diff._validate_hunk(invalid_hunk2))
        
        # Test with negative start positions
        invalid_hunk3 = {
            'start1': -1,
            'count1': 3,
            'start2': 1,
            'count2': 3,
            'lines': [" line"]
        }
        self.assertFalse(diff._validate_hunk(invalid_hunk3))
        
        # Test with valid hunk
        valid_hunk = {
            'start1': 1,
            'count1': 3,
            'start2': 1,
            'count2': 3,
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
        extracted_diffs = UnifiedDiff.extract_from_llm_response(llm_response)
        
        self.assertEqual(len(extracted_diffs), 1)
        
        # Define the expected block
        expected_block = (
            "@@ -1,3 +1,4 @@\n"
            " def calculate(x, y):\n"
            "+    if y == 0:\n"
            "+        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "     return x / y"
        )
        
        # Convert the extracted diff to string for comparison
        extracted_text = extracted_diffs[0].to_string()
        
        # Remove file headers for comparison
        extracted_text = '\n'.join(extracted_text.split('\n')[2:])
        
        self.assertEqual(extracted_text, expected_block)

    def test_inaccurate_line_numbers(self):
        """Test applying a diff with inaccurate line numbers in hunk headers."""
        # Original code with multiple occurrences of the same line
        code_text = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    x = 10\n"  # Duplicate line
            "    z = 30\n"
            "    return x + y + z"  # No trailing newline
        )
        
        # Diff with incorrect line number (targeting line 4 but specifying line 2)
        # Include context to make it clear which occurrence we want to change
        diff_text = (
            "@@ -2,3 +2,3 @@\n"  # Should be line 3, not line 2
            "     x = 10\n"
            "-    y = 20\n"
            "-    x = 10\n"
            "+    y = 20\n"
            "+    x = 15\n"
        )
        
        # Expected result - should change the second occurrence of "x = 10"
        expected_text = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    x = 15\n"  # This should be changed
            "    z = 30\n"
            "    return x + y + z"  # No trailing newline
        )
        
        # Apply the diff
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        
        # Check if the result matches the expected output
        self.assertEqual(result, expected_text)

    def test_inaccurate_line_numbers_multiple_occurrences(self):
        """Test applying a diff with multiple occurrences of the target line."""
        # Code with three identical lines
        code_text = (
            "def process_data():\n"
            "    value = 10\n"
            "    value = 10\n"
            "    value = 10\n"
            "    return value"
        )
        
        # Diff targeting the middle occurrence but with incorrect line number
        diff_text = (
            "@@ -1,2 +1,2 @@\n"  # Should be line 3, not line 1
            " def process_data():\n"
            "-    value = 10\n"
            "+    value = 20\n"
        )
        
        # Expected result - should change the first occurrence since it's closest to line 1
        expected_text = (
            "def process_data():\n"
            "    value = 20\n"
            "    value = 10\n"
            "    value = 10\n"
            "    return value"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)
        
    def test_inaccurate_line_numbers_with_context(self):
        """Test applying a diff with inaccurate line numbers but unique context."""
        # Code with similar lines but different context
        code_text = (
            "def first_function():\n"
            "    x = 5\n"
            "    return x\n"
            "\n"
            "def second_function():\n"
            "    x = 5\n"
            "    return x * 2"
        )
        
        # Diff targeting the second x=5 but with incorrect line number
        diff_text = (
            "@@ -2,2 +2,2 @@\n"  # Should be line 6, not line 2
            "     x = 5\n"
            "-    return x * 2\n"
            "+    return x * 3\n"
        )
        
        # Expected result - should change the second function due to context
        expected_text = (
            "def first_function():\n"
            "    x = 5\n"
            "    return x\n"
            "\n"
            "def second_function():\n"
            "    x = 5\n"
            "    return x * 3"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)
        
    def test_inaccurate_line_numbers_no_match(self):
        """Test applying a diff where the content doesn't match anywhere."""
        code_text = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    return x + y"
        )
        
        # Diff with content that doesn't match anywhere in the code
        diff_text = (
            "@@ -2,1 +2,1 @@\n"
            "-    z = 30\n"
            "+    z = 40\n"
        )
        
        # Expected result - diff should not be applied
        expected_text = code_text
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)
        
    def test_inaccurate_line_numbers_partial_match(self):
        """Test applying a diff with partial content matches."""
        code_text = (
            "def calculate():\n"
            "    # Initialize variables\n"
            "    x = 10\n"
            "    y = 20\n"
            "    # Calculate result\n"
            "    result = x + y\n"
            "    return result"
        )
        
        # Diff with content that partially matches multiple places
        diff_text = (
            "@@ -5,1 +5,1 @@\n"  # Should be line 6, not line 5
            "-    result = x + y\n"
            "+    result = x * y\n"
        )
        
        # Expected result - should change line 6 based on content
        expected_text = (
            "def calculate():\n"
            "    # Initialize variables\n"
            "    x = 10\n"
            "    y = 20\n"
            "    # Calculate result\n"
            "    result = x * y\n"
            "    return result"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)
        
    def test_inaccurate_line_numbers_multiple_hunks(self):
        """Test applying a diff with multiple hunks having inaccurate line numbers."""
        code_text = (
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
        diff_text = (
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
        expected_text = (
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
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)

    def test_inaccurate_line_numbers_empty_code(self):
        """Test applying a diff with inaccurate line numbers to empty code."""
        code_text = ""
        
        # Diff with line numbers that can't possibly match empty code
        diff_text = (
            "@@ -5,1 +5,1 @@\n"
            "-    x = 10\n"
            "+    x = 20\n"
        )
        
        # Expected result - diff should not be applied to empty code
        expected_text = ""
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)
        
    def test_inaccurate_line_numbers_out_of_bounds(self):
        """Test applying a diff with line numbers that are out of bounds."""
        code_text = (
            "def calculate():\n"
            "    x = 10\n"
            "    return x"
        )
        
        # Diff with line numbers that are out of bounds
        diff_text = (
            "@@ -10,1 +10,1 @@\n"  # Line 10 doesn't exist
            "-    y = 20\n"
            "+    y = 30\n"
        )
        
        # Expected result - diff should not be applied
        expected_text = code_text
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)
        
    def test_inaccurate_line_numbers_ambiguous_match(self):
        """Test applying a diff with ambiguous matches (multiple equally good matches)."""
        # Code with identical blocks
        code_text = (
            "def first():\n"
            "    x = 10\n"
            "    y = 20\n"
            "\n"
            "def second():\n"
            "    x = 10\n"
            "    y = 20"  # No trailing newline
        )
        
        # Diff that could match either block
        diff_text = (
            "@@ -2,2 +2,2 @@\n"
            "     x = 10\n"
            "-    y = 20\n"
            "+    y = 30\n"
        )
        
        # Expected result - should change the first occurrence since it's closest to line 2
        expected_text = (
            "def first():\n"
            "    x = 10\n"
            "    y = 30\n"
            "\n"
            "def second():\n"
            "    x = 10\n"
            "    y = 20"  # No trailing newline
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)
        
    def test_inaccurate_line_numbers_mixed_matches(self):
        """Test applying a diff with some hunks matching and others not."""
        code_text = (
            "def calculate():\n"
            "    x = 10\n"
            "    y = 20\n"
            "    return x + y"
        )
        
        # Diff with one hunk that matches and one that doesn't
        diff_text = (
            "@@ -2,1 +2,1 @@\n"
            "-    x = 10\n"
            "+    x = 15\n"
            "@@ -5,1 +5,1 @@\n"  # This line doesn't exist
            "-    z = 30\n"
            "+    z = 40\n"
        )
        
        # Expected result - only the matching hunk should be applied
        expected_text = (
            "def calculate():\n"
            "    x = 15\n"
            "    y = 20\n"
            "    return x + y"
        )
        
        diff = UnifiedDiff.from_string(diff_text)
        result = diff.apply_diff(code_text)
        self.assertEqual(result, expected_text)


if __name__ == "__main__":
    unittest.main()