import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.rewards import categorical_correctness_reward_func


class TestDetectionRewards(unittest.TestCase):
    """Test cases for detection-based reward functions in src/utils/rewards.py."""

    def test_categorical_correctness_reward_func(self):
        """Test the correctness reward function."""
        completions = [
            [{"content": "<answer>CWE-79</answer>"}],
            [{"content": "<answer>CWE-89: SQL Injection</answer>"}]
        ]
        answers = ["CWE-79", "CWE-89"]
        
        # Test exact matches
        rewards = categorical_correctness_reward_func(completions=completions, answers=answers)
        self.assertEqual(rewards, [1.0, 0.0])  # Second one doesn't match exactly
        
        # Test partial matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-89</answer>"}]
        ]
        rewards = categorical_correctness_reward_func(completions=completions, answers=answers)
        self.assertEqual(rewards, [0.0, 1.0])
        
        # Test no matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-90</answer>"}]
        ]
        rewards = categorical_correctness_reward_func(completions=completions, answers=answers)
        self.assertEqual(rewards, [0.0, 0.0])
        
        # Test empty completions
        self.assertEqual(categorical_correctness_reward_func(completions=[], answers=[]), [])


if __name__ == "__main__":
    unittest.main()