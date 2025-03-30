import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rewards import partial_reasoning_format_reward_func, strict_reasoning_format_reward_func


class TestReasoningRewards(unittest.TestCase):
    """Test cases for reasoning-based reward functions in src/utils/rewards.py."""

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
        self.assertAlmostEqual(rewards[2], 0.5, places=1)
    
    def test_reward_function_empty_completions(self):
        """Test reward functions with empty completions list."""
        empty_completions = []
        
        self.assertEqual(partial_reasoning_format_reward_func(empty_completions), [])
        self.assertEqual(strict_reasoning_format_reward_func(empty_completions), [])


if __name__ == "__main__":
    unittest.main() 