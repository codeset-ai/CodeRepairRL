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


if __name__ == "__main__":
    unittest.main()