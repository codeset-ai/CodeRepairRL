import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.rewards import (
    count_search_replace_markers,
    count_unified_diff_markers,
    partial_diff_format_reward_func,
    extract_xml_answer,
    correctness_reward_func,
    strict_format_reward_func,
    count_xml,
    xmlcount_reward_func
)


class TestDiffRewards(unittest.TestCase):
    """Test cases for diff-based reward functions in src/utils/rewards.py."""
    
    # Diff marker counting tests
    
    def test_count_search_replace_markers(self):
        """Test counting of search/replace markers."""
        # Perfect format
        perfect = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "=======\n"
            "def hello_world():\n"
            ">>>>>>> REPLACE"
        )
        self.assertAlmostEqual(count_search_replace_markers(perfect), 0.3, places=1)
        
        # Partial markers
        partial = "SEARCH and ===== but no REPLACE"
        self.assertAlmostEqual(count_search_replace_markers(partial), 0.2, places=1)
        
        # No markers
        none = "No markers at all"
        self.assertAlmostEqual(count_search_replace_markers(none), 0.0, places=1)
    
    def test_count_unified_diff_markers(self):
        """Test counting of unified diff markers."""
        # Perfect format
        perfect = (
            "@@ -1,3 +1,3 @@\n"
            " context\n"
            "-removed\n"
            "+added\n"
        )
        self.assertAlmostEqual(count_unified_diff_markers(perfect), 0.4, places=1)
        
        # Partial markers - only @@ markers
        partial = "@@ but no +/- markers"
        self.assertAlmostEqual(count_unified_diff_markers(partial), 0.1, places=1)
        
        # Only line markers
        line_only = "+added line\n-removed line"
        self.assertAlmostEqual(count_unified_diff_markers(line_only), 0.2, places=1)
        
        # No markers
        none = "No markers at all"
        self.assertAlmostEqual(count_unified_diff_markers(none), 0.0, places=1)
    
    # Diff format reward tests
    
    def test_partial_diff_format_reward_func(self):
        """Test the partial diff format reward function."""
        # Create test completions
        completions = [
            [{"content": "<<<<<<< SEARCH\ndef hello()\n=======\ndef hello_world()\n>>>>>>> REPLACE"}],
            [{"content": "@@ -1,3 +1,3 @@\n+added\n-removed"}],
            [{"content": "No markers at all"}]
        ]
        
        # Test for search_replace diff type
        sr_rewards = partial_diff_format_reward_func(completions, diff_type="search_replace")
        self.assertEqual(len(sr_rewards), 3)
        self.assertAlmostEqual(sr_rewards[0], 0.3, places=1)  # All markers
        self.assertAlmostEqual(sr_rewards[1], 0.0, places=1)  # No search/replace markers
        self.assertAlmostEqual(sr_rewards[2], 0.0, places=1)  # No markers
        
        # Test for unified diff type
        unified_rewards = partial_diff_format_reward_func(completions, diff_type="unified")
        self.assertEqual(len(unified_rewards), 3)
        self.assertAlmostEqual(unified_rewards[0], 0.0, places=1)  # No unified markers
        self.assertGreaterEqual(unified_rewards[1], 0.2)  # Some markers
        self.assertAlmostEqual(unified_rewards[2], 0.0, places=1)  # No markers
    
    # XML handling reward tests
    
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
        perfect_format = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertAlmostEqual(count_xml(perfect_format), 0.5, places=2)
        
        # Missing think opening tag
        missing_think_open = (
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertAlmostEqual(count_xml(missing_think_open), 0.375, places=2)
        
        # Missing think closing tag
        missing_think_close = (
            "<think>\n"
            "Thinking process\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertAlmostEqual(count_xml(missing_think_close), 0.375, places=2)
        
        # Missing answer opening tag
        missing_answer_open = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertAlmostEqual(count_xml(missing_answer_open), 0.375, places=2)
        
        # Missing answer closing tag
        missing_answer_close = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79"
        )
        self.assertAlmostEqual(count_xml(missing_answer_close), 0.375, places=2)
        
        # No tags
        no_tags = "Just some text without any tags"
        self.assertEqual(count_xml(no_tags), 0.0)
        
        # Text before think tag (slight penalty)
        text_before_think = (
            "Some text before\n"
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertLess(count_xml(text_before_think), 0.5)
        
        # Text after answer tag (slight penalty)
        text_after_answer = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>\n"
            "Some text after"
        )
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
    
    # Test that reward functions handle empty completions
    
    def test_reward_function_empty_completions(self):
        """Test reward functions with empty completions list."""
        empty_completions = []
        
        self.assertEqual(partial_diff_format_reward_func(empty_completions, diff_type="search_replace"), [])
        self.assertEqual(xmlcount_reward_func(empty_completions), [])


if __name__ == "__main__":
    unittest.main()