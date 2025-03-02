import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.rewards import (
    extract_xml_answer,
    correctness_reward_func,
    strict_format_reward_func,
    count_xml,
    xmlcount_reward_func,
    # Repair specific reward functions
    count_diff_format,
    partial_diff_format_reward_func,
    diff_similarity_reward,
    diff_similarity_reward_func
)


class TestClassificationRewards(unittest.TestCase):
    """Test cases for reward functions in src/utils/rewards.py."""

    def test_extract_xml_answer(self):
        """Test extracting answers from XML tags."""
        # Valid XML answer
        text_with_answer = "Some text <answer>CWE-79</answer> more text"
        self.assertEqual(extract_xml_answer(text_with_answer), "CWE-79")
        
        # No XML answer
        text_without_answer = "Some text without answer tags"
        self.assertEqual(extract_xml_answer(text_without_answer), "N/A")
        
        # Multiple answers (should extract the first one)
        text_with_multiple = "<answer>First</answer> text <answer>Second</answer>"
        self.assertEqual(extract_xml_answer(text_with_multiple), "First")
        
        # Nested tags (should extract the outer one)
        nested_tags = "Text <answer>Outer <inner>Inner</inner> tag</answer>"
        self.assertEqual(extract_xml_answer(nested_tags), "Outer <inner>Inner</inner> tag")

    def test_correctness_reward_func(self):
        """Test the correctness reward function."""
        # Setup test data
        prompts = ["prompt1", "prompt2"]
        completions = [
            [{"content": "<answer>CWE-79</answer>"}],
            [{"content": "<answer>CWE-89: SQL Injection</answer>"}]
        ]
        answers = ["CWE-79", "CWE-89"]
        
        # Test exact matches
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [2.0, 0.5])
        
        # Test partial matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-89</answer>"}]
        ]
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [0.0, 2.0])
        
        # Test no matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-90</answer>"}]
        ]
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [0.0, 0.0])

    def test_strict_format_reward_func(self):
        """Test the strict format reward function."""
        # Valid format
        valid_completion = [
            {"content": "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"}
        ]
        rewards = strict_format_reward_func([valid_completion])
        self.assertEqual(rewards, [0.5])
        
        # Invalid format - missing think tag
        invalid_completion1 = [
            {"content": "Thinking process\n<answer>\nCWE-79\n</answer>"}
        ]
        rewards = strict_format_reward_func([invalid_completion1])
        self.assertEqual(rewards, [0.0])
        
        # Invalid format - missing answer tag
        invalid_completion2 = [
            {"content": "<think>\nThinking process\n</think>\nCWE-79"}
        ]
        rewards = strict_format_reward_func([invalid_completion2])
        self.assertEqual(rewards, [0.0])
        
        # Multiple completions
        completions = [
            valid_completion,
            invalid_completion1,
            invalid_completion2
        ]
        rewards = strict_format_reward_func(completions)
        self.assertEqual(rewards, [0.5, 0.0, 0.0])

    def test_count_xml(self):
        """Test the count_xml function."""
        # Perfect format
        perfect_format = "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"
        self.assertAlmostEqual(count_xml(perfect_format), 0.5, places=2)
        
        # Missing think opening tag
        missing_think_open = "Thinking process\n</think>\n<answer>\nCWE-79\n</answer>"
        self.assertAlmostEqual(count_xml(missing_think_open), 0.375, places=2)
        
        # Missing think closing tag
        missing_think_close = "<think>\nThinking process\n<answer>\nCWE-79\n</answer>"
        self.assertAlmostEqual(count_xml(missing_think_close), 0.375, places=2)
        
        # Missing answer opening tag
        missing_answer_open = "<think>\nThinking process\n</think>\nCWE-79\n</answer>"
        self.assertAlmostEqual(count_xml(missing_answer_open), 0.375, places=2)
        
        # Missing answer closing tag
        missing_answer_close = "<think>\nThinking process\n</think>\n<answer>\nCWE-79"
        self.assertAlmostEqual(count_xml(missing_answer_close), 0.375, places=2)
        
        # No tags
        no_tags = "Just some text without any tags"
        self.assertEqual(count_xml(no_tags), 0.0)
        
        # Text before think tag (slight penalty)
        text_before_think = "Some text before\n<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"
        self.assertLess(count_xml(text_before_think), 0.5)
        
        # Text after answer tag (slight penalty)
        text_after_answer = "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>\nSome text after"
        self.assertLess(count_xml(text_after_answer), 0.5)

    def test_xmlcount_reward_func(self):
        """Test the xmlcount_reward_func function."""
        completions = [
            [{"content": "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"}],
            [{"content": "Just some text without any tags"}],
            [{"content": "<think>\nThinking process\n</think>\nCWE-79"}]
        ]
        
        rewards = xmlcount_reward_func(completions)
        self.assertAlmostEqual(rewards[0], 0.5, places=2)
        self.assertEqual(rewards[1], 0.0)
        self.assertLess(rewards[2], 0.5)


class TestRepairRewards(unittest.TestCase):
    def test_count_diff_format(self):
        """Test the count of diff format markers."""
        # Valid diff format
        valid_diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertAlmostEqual(count_diff_format(valid_diff), 0.3, places=5)
        
        # Missing SEARCH marker
        missing_search = (
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertAlmostEqual(count_diff_format(missing_search), 0.2, places=5)
        
        # Missing divider
        missing_divider = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertAlmostEqual(count_diff_format(missing_divider), 0.2, places=5)
        
        # Missing REPLACE marker
        missing_replace = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
        )
        self.assertAlmostEqual(count_diff_format(missing_replace), 0.2, places=5)
        
        # Has all markers but in wrong order
        wrong_order = (
            ">>>>>>> REPLACE\n"
            "def hello():\n"
            "    print('hello world')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello')\n"
            "<<<<<<< SEARCH"
        )
        self.assertAlmostEqual(count_diff_format(wrong_order), 0.3, places=5)
        
        # No markers
        no_markers = "This is just some random text"
        self.assertAlmostEqual(count_diff_format(no_markers), 0.0, places=5)
        
        # Mixed valid and invalid blocks
        mixed_blocks = (
            ">>>>>>> REPLACE\n"
            "def wrong():\n"
            "    print('wrong')\n"
            "=======\n"
            "<<<<<<< SEARCH\n\n"
            "<<<<<<< SEARCH\n"
            "def correct():\n"
            "    print('correct')\n"
            "=======\n"
            "def correct():\n"
            "    print('correct fixed')\n"
            ">>>>>>> REPLACE"
        )
        self.assertAlmostEqual(count_diff_format(mixed_blocks), 0.3, places=5)

    def test_partial_diff_format_reward_func(self):
        """Test the partial_diff_format_reward_func function."""
        completions = [
            [{"content": "<<<<<<< SEARCH\ndef hello():\n    print('hello')\n=======\ndef hello():\n    print('hello world')\n>>>>>>> REPLACE"}],
            [{"content": "def hello():\n    print('hello')\n=======\ndef hello():\n    print('hello world')\n>>>>>>> REPLACE"}],
            [{"content": "This is just some random text"}]
        ]
        
        rewards = partial_diff_format_reward_func(completions)
        self.assertAlmostEqual(rewards[0], 0.3, places=5)  # Valid diff
        self.assertAlmostEqual(rewards[1], 0.2, places=5)  # Missing SEARCH marker
        self.assertAlmostEqual(rewards[2], 0.0, places=5)  # No markers

    def test_diff_similarity_reward(self):
        """Test the diff_similarity_reward function."""
        # Identical diffs
        reference_diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertEqual(diff_similarity_reward(reference_diff, reference_diff), 1.0)
        
        # Similar diffs (minor differences)
        similar_diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello, world!')\n"  # Slight difference
            ">>>>>>> REPLACE"
        )
        similarity = diff_similarity_reward(reference_diff, similar_diff)
        self.assertGreater(similarity, 0.8)  # Should be high but not 1.0
        
        # Very different diffs
        different_diff = (
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        similarity = diff_similarity_reward(reference_diff, different_diff)
        self.assertLess(similarity, 0.5)  # Should be low
        
        # Invalid diff format
        invalid_diff = "This is not a valid diff"
        self.assertEqual(diff_similarity_reward(reference_diff, invalid_diff), 0.0)

    def test_diff_similarity_reward_func(self):
        """Test the diff_similarity_reward_func function."""
        reference_diff = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        reference_diffs = [reference_diff] * 4
        
        completions = [
            [{"content": (
                "<think>\n"
                "This is a test\n"
                "</think>\n"
                "<answer>\n"
                "```python\n"
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print('hello')\n"
                "=======\n"
                "def hello():\n"
                "    print('hello world')\n"
                ">>>>>>> REPLACE\n"
                "```\n"
                "</answer>\n"
            )}],
            [{"content": (
                "<think>\n"
                "This is a test\n"
                "</think>\n"
                "<answer>\n"
                "```python\n"
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print('hello')\n"
                "=======\n"
                "def hello():\n"
                "    print('goodbye world')\n"  # Slight difference
                ">>>>>>> REPLACE\n"
                "```\n"
                "</answer>\n"
            )}],
            [{"content": (
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print('hello')\n"
                "=======\n"
                "def hello():\n"
                "    print('hello world')\n"
                ">>>>>>> REPLACE"
            )}],
            [{"content": "This is not a valid diff"}]
        ]
        
        rewards = diff_similarity_reward_func(completions, reference_diffs)
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)  # Identical to reference
        self.assertGreater(rewards[1], 0.8)  # Highly similar to reference
        self.assertEqual(rewards[2], 0.0)  # Missing <answer> tags
        self.assertEqual(rewards[3], 0.0)  # Invalid diff format


if __name__ == "__main__":
    unittest.main() 