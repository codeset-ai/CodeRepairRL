import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diff import SearchReplaceDiff, get_diff


class TestSearchReplaceDiff(unittest.TestCase):
    """Test cases for SearchReplaceDiff in src/utils/diff.py."""
    
    def setUp(self):
        """Set up a SearchReplaceDiff instance for testing."""
        self.diff = SearchReplaceDiff()

    # Basic parsing and functionality tests
    
    def test_parse_search_replace_block(self):
        """Test parsing a single search/replace block."""
        block = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        search_content, replace_content = self.diff.parse_block(block)
        
        self.assertEqual(search_content, "def hello():\n    print('hello')")
        self.assertEqual(replace_content, "def hello():\n    print('hello world')")

    def test_parse_search_replace_block_missing_divider(self):
        """Test parsing a search/replace block with a missing divider."""
        broken_block = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            # missing divider
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "def hello():\n"
        )

        search_content, replace_content = self.diff.parse_block(broken_block)

        self.assertIsNone(search_content)
        self.assertIsNone(replace_content)

    def test_parse_search_replace_block_invalid_block(self):
        """Test parsing an invalid search/replace block."""
        invalid_block = "This is not a valid SEARCH/REPLACE block"
        search_content, replace_content = self.diff.parse_block(invalid_block)
        
        self.assertIsNone(search_content)
        self.assertIsNone(replace_content)

    def test_parse_search_replace_diff(self):
        """Test parsing a complete search/replace diff."""
        diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        replacements = self.diff.parse_diff(diff)
        
        self.assertEqual(len(replacements), 2)
        self.assertEqual(replacements[0][0], "def hello():\n    print('hello')")
        self.assertEqual(replacements[0][1], "def hello():\n    print('hello world')")
        self.assertEqual(replacements[1][0], "def goodbye():\n    print('goodbye')")
        self.assertEqual(replacements[1][1], "def goodbye():\n    print('goodbye world')")

    def test_parse_search_replace_diff_empty_diff(self):
        """Test parsing an empty diff."""
        self.assertEqual(self.diff.parse_diff(""), [])

    def test_apply_search_replace_diff(self):
        """Test applying a search/replace diff to code."""
        code = "def hello():\n    print('hello')\n\ndef goodbye():\n    print('goodbye')"
        diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        result = self.diff.apply_diff(code, diff)
        expected = "def hello():\n    print('hello world')\n\ndef goodbye():\n    print('goodbye world')"
        
        self.assertEqual(result, expected)
        
    def test_apply_search_replace_diff_empty_diff(self):
        """Test that applying an empty diff returns the original code."""
        code = "def hello():\n    print('hello')\n\ndef goodbye():\n    print('goodbye')"
        self.assertEqual(self.diff.apply_diff(code, ""), code)

    def test_apply_search_replace_diff_invalid_diff(self):
        """Test that applying an invalid diff returns the original code."""
        code = "def hello():\n    print('hello')\n\ndef goodbye():\n    print('goodbye')"
        invalid_diff = "This is not a valid SEARCH/REPLACE diff"
        self.assertEqual(self.diff.apply_diff(code, invalid_diff), code)

    def test_apply_search_replace_diff_to_empty_code(self):
        """Test that applying a diff to empty code returns the diff."""
        diff = (
            "<<<<<<< SEARCH\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertEqual(self.diff.apply_diff("", diff), "def hello():\n    print('hello world')")
    

    def test_validate_search_replace_diff(self):
        """Test validating search/replace diffs."""
        # Valid diff
        valid_diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        # Invalid diffs
        missing_search = (
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        missing_divider = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            ">>>>>>> REPLACE"
        )
        
        missing_replace = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')"
        )
        
        wrong_order = (
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            ">>>>>>> REPLACE"
        )
        
        self.assertTrue(self.diff.is_valid_format(valid_diff))
        self.assertFalse(self.diff.is_valid_format(missing_search))
        self.assertFalse(self.diff.is_valid_format(missing_divider))
        self.assertFalse(self.diff.is_valid_format(missing_replace))
        self.assertFalse(self.diff.is_valid_format(wrong_order))
        
        # Empty diff is valid
        self.assertTrue(self.diff.is_valid_format(""))

    def test_extract_search_replace_blocks_from_llm_response(self):
        """Test extracting search/replace blocks from an LLM response."""
        llm_response = (
            "<think>\n"
            "The main issue is that the function doesn't handle negative inputs correctly.\n"
            "Also, there's a potential division by zero error.\n"
            "</think>\n"
            "<answer>\n"
            "Here are the fixes:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "def calculate(x, y):\n"
            "    return x / y\n"
            "=======\n"
            "def calculate(x, y):\n"
            "    if y == 0:\n"
            "        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "    return x / y\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "\n"
            "And another issue:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "    return value + 10\n"
            "=======\n"
            "    if value < 0:\n"
            "        value = 0\n"
            "    return value + 10\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "</answer>"
        )
        
        # Extract the blocks
        extracted = self.diff.extract_from_llm_response(llm_response)
        
        # Define the expected blocks
        expected_blocks = (
            "<<<<<<< SEARCH\n"
            "def calculate(x, y):\n"
            "    return x / y\n"
            "=======\n"
            "def calculate(x, y):\n"
            "    if y == 0:\n"
            "        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "    return x / y\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "    return value + 10\n"
            "=======\n"
            "    if value < 0:\n"
            "        value = 0\n"
            "    return value + 10\n"
            ">>>>>>> REPLACE"
        )

        self.assertEqual(extracted, expected_blocks)

    def test_extract_search_replace_blocks_from_llm_response_no_tags(self):
        """Test extracting search/replace blocks from an LLM response with no tags."""
        llm_response = (
            "Here are the fixes:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "def calculate(x, y):\n"
            "    return x / y\n"
            "=======\n"
            "def calculate(x, y):\n"
            "    if y == 0:\n"
            "        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "    return x / y\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "\n"
            "And another issue:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "    return value + 10\n"
            "=======\n"
            "    if value < 0:\n"
            "        value = 0\n"
            "    return value + 10\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        
        # Extract the blocks
        extracted = self.diff.extract_from_llm_response(llm_response)
        
        # Define the expected blocks
        expected_blocks = (
            "<<<<<<< SEARCH\n"
            "def calculate(x, y):\n"
            "    return x / y\n"
            "=======\n"
            "def calculate(x, y):\n"
            "    if y == 0:\n"
            "        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "    return x / y\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "    return value + 10\n"
            "=======\n"
            "    if value < 0:\n"
            "        value = 0\n"
            "    return value + 10\n"
            ">>>>>>> REPLACE"
        )

        self.assertEqual(extracted, expected_blocks)

    def test_extract_search_replace_blocks_from_llm_response_no_blocks(self):
        """Test extracting search/replace blocks from an LLM response with no blocks."""
        no_blocks_response = "The LLM is yapping here without following the instructions."
        self.assertEqual(self.diff.extract_from_llm_response(no_blocks_response), "")
        
    def test_extract_search_replace_blocks_without_code_fences(self):
        """Test extracting search/replace blocks from an LLM response without code fences."""
        llm_response = (
            "Here's my fix:\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def calculate(x, y):\n"
            "    return x / y\n"
            "=======\n"
            "def calculate(x, y):\n"
            "    if y == 0:\n"
            "        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "    return x / y\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "This should handle the division by zero error."
        )
        
        # Extract the blocks
        extracted = self.diff.extract_from_llm_response(llm_response)
        
        # Define the expected block
        expected_block = (
            "<<<<<<< SEARCH\n"
            "def calculate(x, y):\n"
            "    return x / y\n"
            "=======\n"
            "def calculate(x, y):\n"
            "    if y == 0:\n"
            "        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "    return x / y\n"
            ">>>>>>> REPLACE"
        )
        
        self.assertEqual(extracted, expected_block)

    # Diff generation tests
    def test_simple_replace(self):
        """Test a simple replacement of a single line."""
        before_code = "def hello():\n    print('hello')\n    return None"
        after_code = "def hello():\n    print('hello world')\n    return None"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "    print('hello')\n"
            "=======\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_multi_line_replace(self):
        """Test replacement of multiple consecutive lines."""
        before_code = "def calculate(x, y):\n    # Add two numbers\n    result = x + y\n    return result"
        after_code = "def calculate(x, y):\n    # Add two numbers and multiply by 2\n    result = (x + y) * 2\n    return result"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "=======\n"
            "    # Add two numbers and multiply by 2\n"
            "    result = (x + y) * 2\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_simple_deletion(self):
        """Test deletion of a line."""
        before_code = "def process():\n    x = 1\n    # This is a comment\n    y = 2\n    return x + y"
        after_code = "def process():\n    x = 1\n    y = 2\n    return x + y"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "    # This is a comment\n"
            "=======\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_multi_line_deletion(self):
        """Test deletion of multiple consecutive lines."""
        before_code = "def example():\n    # Step 1\n    # Step 2\n    # Step 3\n    return True"
        after_code = "def example():\n    return True"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "    # Step 1\n"
            "    # Step 2\n"
            "    # Step 3\n"
            "=======\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_simple_insertion(self):
        """Test insertion of a single line."""
        before_code = "def greet(name):\n    return f'Hello, {name}!'"
        after_code = "def greet(name):\n    name = name.strip()\n    return f'Hello, {name}!'"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "def greet(name):\n"
            "=======\n"
            "def greet(name):\n"
            "    name = name.strip()\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_multi_line_insertion(self):
        """Test insertion of multiple consecutive lines."""
        before_code = "def validate(data):\n    return True"
        after_code = "def validate(data):\n    if not data:\n        return False\n    if not isinstance(data, dict):\n        return False\n    return True"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "def validate(data):\n"
            "=======\n"
            "def validate(data):\n"
            "    if not data:\n"
            "        return False\n"
            "    if not isinstance(data, dict):\n"
            "        return False\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_multiple_changes(self):
        """Test multiple separate changes in the same code."""
        before_code = (
            "def process_data(items):\n"
            "    results = []\n"
            "    for item in items:\n"
            "        # Process item\n"
            "        results.append(item * 2)\n"
            "    return results"
        )
        after_code = (
            "def process_data(items):\n"
            "    if not items:\n"
            "        return []\n"
            "    results = []\n"
            "    for item in items:\n"
            "        # Process and transform item\n"
            "        results.append(item * 2 + 1)\n"
            "    return results"
        )
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "def process_data(items):\n"
            "=======\n"
            "def process_data(items):\n"
            "    if not items:\n"
            "        return []\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "        # Process item\n"
            "        results.append(item * 2)\n"
            "=======\n"
            "        # Process and transform item\n"
            "        results.append(item * 2 + 1)\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_indentation_preservation(self):
        """Test that indentation is properly preserved in the diff."""
        before_code = "def nested():\n    if True:\n        for i in range(10):\n            print(i)\n    return"
        after_code = "def nested():\n    if True:\n        for i in range(10):\n            if i % 2 == 0:\n                print(i)\n    return"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "            print(i)\n"
            "=======\n"
            "            if i % 2 == 0:\n"
            "                print(i)\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_empty_before_code(self):
        """Test with empty before_code (complete insertion)."""
        before_code = ""
        after_code = "def new_function():\n    return 'Hello, world!'"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "=======\n"
            "def new_function():\n"
            "    return 'Hello, world!'\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_empty_after_code(self):
        """Test with empty after_code (complete deletion)."""
        before_code = "def old_function():\n    return 'Goodbye, world!'"
        after_code = ""
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "def old_function():\n"
            "    return 'Goodbye, world!'\n"
            "=======\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_no_changes(self):
        """Test with identical before and after code (no changes)."""
        code = "def unchanged():\n    return 42"
        
        actual_diff = self.diff.generate_diff(code, code)
        self.assertEqual(actual_diff, "")

    def test_whitespace_changes(self):
        """Test with only whitespace changes."""
        before_code = "def whitespace():\n    x = 1\n    y = 2\n    return x + y"
        after_code = "def whitespace():\n    x = 1\n    y = 2\n    return x+y"  # Removed space around +
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "    return x + y\n"
            "=======\n"
            "    return x+y\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

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
    
    def test_parse_malformed_markers(self):
        """Test parsing a block with malformed markers but recoverable content."""
        block = (
            "<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>> REPLACE"
        )
        
        search_content, replace_content = self.diff.parse_block(block)
        
        self.assertEqual(search_content, "def hello():\n    print('hello')")
        self.assertEqual(replace_content, "def hello():\n    print('hello world')")

    def test_parse_excessive_markers(self):
        """Test parsing a block with excessive markers."""
        block = (
            "<<<<<<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "===========\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>>>>>> REPLACE"
        )
        
        search_content, replace_content = self.diff.parse_block(block)
        
        self.assertEqual(search_content, "def hello():\n    print('hello')")
        self.assertEqual(replace_content, "def hello():\n    print('hello world')")
        
    def test_parse_whitespace_in_markers(self):
        """Test parsing a block with whitespace in markers."""
        block = (
            "<<<<<<< SEARCH \n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE "
        )
        
        search_content, replace_content = self.diff.parse_block(block)
        
        self.assertEqual(search_content, "def hello():\n    print('hello')")
        self.assertEqual(replace_content, "def hello():\n    print('hello world')")
        
    def test_parse_very_malformed_but_recoverable(self):
        """Test parsing a block that's very malformed but still has the key parts."""
        block = (
            "< SEARCH >\n"
            "def hello():\n"
            "    print('hello')\n"
            "=====\n"
            "def hello():\n"
            "    print('hello world')\n"
            "> REPLACE <"
        )
        
        search_content, replace_content = self.diff.parse_block(block)
        
        self.assertEqual(search_content, "def hello():\n    print('hello')")
        self.assertEqual(replace_content, "def hello():\n    print('hello world')")
    
    def test_parse_completely_invalid(self):
        """Test parsing a completely invalid block."""
        block = "This is not a diff at all."
        
        search_content, replace_content = self.diff.parse_block(block)
        
        self.assertIsNone(search_content)
        self.assertIsNone(replace_content)
    
    def test_quality_validation_perfect(self):
        """Test quality validation on a perfect diff."""
        diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        quality = self.diff.validate_quality(diff)
        self.assertEqual(quality, 1.0)
    
    def test_quality_validation_good_enough(self):
        """Test quality validation on a good enough diff."""
        diff = (
            "<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=====\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>> REPLACE"
        )
        
        quality = self.diff.validate_quality(diff)
        self.assertGreaterEqual(quality, 0.7)
        self.assertLessEqual(quality, 0.9)
    
    def test_quality_validation_recoverable(self):
        """Test quality validation on a recoverable diff."""
        diff = (
            "<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "===\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>> REPLACE"
        )
        
        quality = self.diff.validate_quality(diff)
        # This is a fairly recoverable diff with all markers
        self.assertGreaterEqual(quality, 0.4)
    
    def test_quality_validation_poor(self):
        """Test quality validation on a poor diff."""
        diff = (
            "Here's the diff: SEARCH and hello() and ===== and REPLACE"
        )
        
        quality = self.diff.validate_quality(diff)
        self.assertLessEqual(quality, 0.3)
    
    def test_quality_validation_invalid(self):
        """Test quality validation on an invalid diff."""
        diff = "This is not a diff at all."
        
        quality = self.diff.validate_quality(diff)
        self.assertLess(quality, 0.2)

    def test_safe_apply_perfect_diff(self):
        """Test safely applying a perfect diff."""
        code = "def hello():\n    print('hello')\n    return"
        diff = (
            "<<<<<<< SEARCH\n"
            "    print('hello')\n"
            "=======\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        result, quality = self.diff.safe_apply_diff(code, diff)
        
        self.assertEqual(result, "def hello():\n    print('hello world')\n    return")
        self.assertEqual(quality, 1.0)

    def test_safe_apply_recoverable_diff(self):
        """Test safely applying a recoverable diff."""
        code = "def hello():\n    print('hello')\n    return"
        diff = (
            "<<<< SEARCH\n"
            "    print('hello')\n"
            "====\n"
            "    print('hello world')\n"
            ">>>> REPLACE"
        )
        
        result, quality = self.diff.safe_apply_diff(code, diff)
        
        # The quality should be good enough to apply
        self.assertGreaterEqual(quality, 0.4)
        
        # If the diff was applied, check that it was applied correctly
        if result != code:
            self.assertEqual(result, "def hello():\n    print('hello world')\n    return")

    def test_safe_apply_invalid_diff(self):
        """Test safely applying an invalid diff."""
        code = "def hello():\n    print('hello')\n    return"
        diff = "This is not a diff at all."
        
        result, quality = self.diff.safe_apply_diff(code, diff)
        
        # Invalid diff should return original code
        self.assertEqual(result, code)
        self.assertLess(quality, 0.2)
        
    def test_lenient_validation(self):
        """Test lenient validation on different diff formats."""
        valid_strict = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "=======\n"
            "def hello_world():\n"
            ">>>>>>> REPLACE"
        )
        
        valid_lenient = (
            "<< SEARCH\n"
            "def hello():\n"
            "==\n"
            "def hello_world():\n"
            ">> REPLACE"
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
    
    def test_comment_only_change(self):
        """Test changes that only affect comments."""
        before_code = "def add(a, b):\n    # Add two numbers\n    return a + b"
        after_code = "def add(a, b):\n    # Add two numbers together\n    return a + b"
        
        expected_diff = (
            "<<<<<<< SEARCH\n"
            "    # Add two numbers\n"
            "=======\n"
            "    # Add two numbers together\n"
            ">>>>>>> REPLACE"
        )
        
        actual_diff = self.diff.generate_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)
        
    def test_apply_diff_with_surrounding_context(self):
        """Test applying a diff with surrounding context that matches multiple locations."""
        # Code with repeated sections
        code = (
            "def repeat():\n"
            "    # Block 1\n"
            "    x = 1\n"
            "    print(x)\n"
            "    \n"
            "    # Block 2\n"
            "    x = 1\n"
            "    print(x)\n"
            "    \n"
            "    # Block 3\n"
            "    x = 1\n"
            "    print(x)\n"
        )
        
        # Diff targeting the second block specifically
        diff = (
            "<<<<<<< SEARCH\n"
            "    # Block 2\n"
            "    x = 1\n"
            "    print(x)\n"
            "=======\n"
            "    # Block 2 (modified)\n"
            "    x = 2\n"
            "    print(x * 2)\n"
            ">>>>>>> REPLACE"
        )
        
        expected = (
            "def repeat():\n"
            "    # Block 1\n"
            "    x = 1\n"
            "    print(x)\n"
            "    \n"
            "    # Block 2 (modified)\n"
            "    x = 2\n"
            "    print(x * 2)\n"
            "    \n"
            "    # Block 3\n"
            "    x = 1\n"
            "    print(x)\n"
        )
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_partial_match_with_specific_replacement(self):
        """Test applying a diff where the search string is only a partial match."""
        code = "# Important function\ndef calculate():\n    return 1 + 2 + 3  # Returns 6"
        diff = (
            "<<<<<<< SEARCH\n"
            "    return 1 + 2 + 3\n"
            "=======\n"
            "    return (1 + 2 + 3) * 2\n"
            ">>>>>>> REPLACE"
        )
        
        # The diff should still apply even though it's not matching the comment
        expected = "# Important function\ndef calculate():\n    return (1 + 2 + 3) * 2  # Returns 6"
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
    
    def test_multiple_diffs_with_same_search(self):
        """Test handling of multiple diff blocks with the same search pattern."""
        code = "x = 1\nx = 1\nx = 1"
        diff = (
            "<<<<<<< SEARCH\n"
            "x = 1\n"
            "=======\n"
            "x = 2\n"
            ">>>>>>> REPLACE"
        )
        
        # All instances should be replaced
        expected = "x = 2\nx = 2\nx = 2"
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_empty_search_content_with_nonempty_code(self):
        """Test handling of empty search content with non-empty original code."""
        code = "def existing_function():\n    return 42"
        diff = (
            "<<<<<<< SEARCH\n"
            "=======\n"
            "def new_function():\n"
            "    return 100\n"
            ">>>>>>> REPLACE"
        )
        
        # Empty search should NOT insert between every character when code is non-empty
        expected = "def existing_function():\n    return 42"
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)
        
    def test_empty_search_content_with_empty_code(self):
        """Test handling of empty search content with empty original code (new file creation)."""
        code = ""
        diff = (
            "<<<<<<< SEARCH\n"
            "=======\n"
            "def new_function():\n"
            "    return 100\n"
            ">>>>>>> REPLACE"
        )
        
        # Empty search with empty code should create a new file
        expected = "def new_function():\n    return 100"
        
        result = self.diff.apply_diff(code, diff)
        self.assertEqual(result, expected)

    def test_alternative_block_separators_triple_newline(self):
        """Test parsing diff blocks separated by triple newlines."""
        diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "\n\n"  # Triple newline separator
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        parsed = self.diff.parse_diff(diff)
        
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0][0], "def hello():\n    print('hello')")
        self.assertEqual(parsed[0][1], "def hello():\n    print('hello world')")
        self.assertEqual(parsed[1][0], "def goodbye():\n    print('goodbye')")
        self.assertEqual(parsed[1][1], "def goodbye():\n    print('goodbye world')")
        
    def test_no_block_separators(self):
        """Test parsing diff blocks with no clear separators."""
        diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        # Add a newline between blocks to make it parse correctly
        diff_with_separator = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        # Test with the original diff (no separator)
        parsed = self.diff.parse_diff(diff)
        # The current implementation treats this as a single block
        self.assertEqual(len(parsed), 1)
        
        # Test with a proper separator
        parsed_with_separator = self.diff.parse_diff(diff_with_separator)
        # This should parse as two blocks
        self.assertEqual(len(parsed_with_separator), 2)


if __name__ == "__main__":
    unittest.main()