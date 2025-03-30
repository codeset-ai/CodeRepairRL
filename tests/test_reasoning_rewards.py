import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rewards import (
    extract_xml_answer,
    count_xml,
    partial_reasoning_format_reward_func,
    strict_reasoning_format_reward_func
)


class TestReasoningRewards(unittest.TestCase):
    """Test cases for reasoning-based reward functions in src/utils/rewards.py."""
    
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
    
    def test_count_xml(self):
        """Test the count_xml function."""
        # Perfect format
        perfect_format = "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"
        self.assertAlmostEqual(count_xml(perfect_format), 1.0, places=2)
        
        # Missing think tag
        missing_think = "Thinking process\n<answer>\nCWE-79\n</answer>"
        self.assertAlmostEqual(count_xml(missing_think), 0.5, places=1)
        
        # Missing answer tag
        missing_answer = "<think>\nThinking process\n</think>\nCWE-79"
        self.assertAlmostEqual(count_xml(missing_answer), 0.5, places=1)

        # No tags
        no_tags = "Just some text without any tags"
        self.assertEqual(count_xml(no_tags), 0.0)

    def test_strict_reasoning_format_reward_func(self):
        """Test the strict reasoning format reward function."""
        # Valid format
        valid_completion = [
            [{"content": "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"}]
        ]
        rewards = strict_reasoning_format_reward_func(valid_completion)
        self.assertEqual(rewards, [1.0])
        
        # Invalid format - missing think tag
        invalid_completion1 = [
            [{"content": "Thinking process\n<answer>\nCWE-79\n</answer>"}]
        ]
        rewards = strict_reasoning_format_reward_func(invalid_completion1)
        self.assertEqual(rewards, [0.0])
        
        # Invalid format - missing answer tag
        invalid_completion2 = [
            [{"content": "<think>\nThinking process\n</think>\nCWE-79"}]
        ]
        rewards = strict_reasoning_format_reward_func(invalid_completion2)
        self.assertEqual(rewards, [0.0])
        
        # Multiple completions
        completions = [
            [{"content": "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"}],
            [{"content": "Thinking process\n<answer>\nCWE-79\n</answer>"}],
            [{"content": "<think>\nThinking process\n</think>\nCWE-79"}]
        ]
        rewards = strict_reasoning_format_reward_func(completions)
        self.assertEqual(rewards, [1.0, 0.0, 0.0])

    def test_partial_reasoning_format_reward_func(self):
        """Test the partial reasoning format reward function."""
        completions = [
            [{"content": "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"}],
            [{"content": "Just some text without any tags"}],
            [{"content": "<think>\nThinking process\n</think>\nCWE-79"}]
        ]
        
        rewards = partial_reasoning_format_reward_func(completions)
        self.assertAlmostEqual(rewards[0], 1.0, places=2)
        self.assertEqual(rewards[1], 0.0)
        self.assertLess(rewards[2], 1.0)
    
    def test_reward_function_empty_completions(self):
        """Test reward functions with empty completions list."""
        empty_completions = []
        
        self.assertEqual(partial_reasoning_format_reward_func(empty_completions), [])
        self.assertEqual(strict_reasoning_format_reward_func(empty_completions), [])


if __name__ == "__main__":
    unittest.main() 