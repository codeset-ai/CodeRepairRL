import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diff import (
    parse_search_replace_block,
    parse_search_replace_diff,
    apply_search_replace_diff,
    is_valid_diff_format,
    extract_search_replace_blocks_from_llm_response,
    generate_search_replace_diff,
)


class TestDiffUtils(unittest.TestCase):
    """Test cases for diff utilities in src/utils/diff.py."""

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
        
        search_content, replace_content = parse_search_replace_block(block)
        
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

        search_content, replace_content = parse_search_replace_block(broken_block)

        self.assertIsNone(search_content)
        self.assertIsNone(replace_content)

    def test_parse_search_replace_block_invalid_block(self):
        """Test parsing an invalid search/replace block."""
        invalid_block = "This is not a valid SEARCH/REPLACE block"
        search_content, replace_content = parse_search_replace_block(invalid_block)
        
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
        
        replacements = parse_search_replace_diff(diff)
        
        self.assertEqual(len(replacements), 2)
        self.assertEqual(replacements[0][0], "def hello():\n    print('hello')")
        self.assertEqual(replacements[0][1], "def hello():\n    print('hello world')")
        self.assertEqual(replacements[1][0], "def goodbye():\n    print('goodbye')")
        self.assertEqual(replacements[1][1], "def goodbye():\n    print('goodbye world')")

    def test_parse_search_replace_diff_empty_diff(self):
        """Test parsing an empty diff."""
        self.assertEqual(parse_search_replace_diff(""), [])

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
        
        result = apply_search_replace_diff(code, diff)
        expected = "def hello():\n    print('hello world')\n\ndef goodbye():\n    print('goodbye world')"
        
        self.assertEqual(result, expected)
        
    def test_apply_search_replace_diff_empty_diff(self):
        """Test that applying an empty diff returns the original code."""
        code = "def hello():\n    print('hello')\n\ndef goodbye():\n    print('goodbye')"
        self.assertEqual(apply_search_replace_diff(code, ""), code)

    def test_apply_search_replace_diff_invalid_diff(self):
        """Test that applying an invalid diff returns the original code."""
        code = "def hello():\n    print('hello')\n\ndef goodbye():\n    print('goodbye')"
        invalid_diff = "This is not a valid SEARCH/REPLACE diff"
        self.assertEqual(apply_search_replace_diff(code, invalid_diff), code)

    def test_apply_search_replace_diff_to_empty_code(self):
        """Test that applying a diff to empty code returns the diff."""
        diff = (
            "<<<<<<< SEARCH\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertEqual(apply_search_replace_diff("", diff), "def hello():\n    print('hello world')")
    

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
        
        self.assertTrue(is_valid_diff_format(valid_diff))
        self.assertFalse(is_valid_diff_format(missing_search))
        self.assertFalse(is_valid_diff_format(missing_divider))
        self.assertFalse(is_valid_diff_format(missing_replace))
        self.assertFalse(is_valid_diff_format(wrong_order))
        
        # Empty diff is valid
        self.assertTrue(is_valid_diff_format(""))

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
        extracted = extract_search_replace_blocks_from_llm_response(llm_response)
        
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
        extracted = extract_search_replace_blocks_from_llm_response(llm_response)
        
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
        self.assertEqual(extract_search_replace_blocks_from_llm_response(no_blocks_response), "")

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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
        self.assertEqual(actual_diff, expected_diff)

    def test_no_changes(self):
        """Test with identical before and after code (no changes)."""
        code = "def unchanged():\n    return 42"
        
        actual_diff = generate_search_replace_diff(code, code)
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
        
        actual_diff = generate_search_replace_diff(before_code, after_code)
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
        diff = generate_search_replace_diff(before_code, after_code)
        
        # Apply diff
        result = apply_search_replace_diff(before_code, diff)
        
        # Verify result matches after_code
        self.assertEqual(result, after_code)


if __name__ == "__main__":
    unittest.main() 