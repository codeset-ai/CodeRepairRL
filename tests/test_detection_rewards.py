import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.rewards import (
    extract_xml_answer,
    correctness_reward_func
)


class TestDetectionRewards(unittest.TestCase):
    """Test cases for detection-based reward functions in src/utils/rewards.py."""

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
        completions = [
            [{"content": "<answer>CWE-79</answer>"}],
            [{"content": "<answer>CWE-89: SQL Injection</answer>"}]
        ]
        answers = ["CWE-79", "CWE-89"]
        
        # Test exact matches
        rewards = correctness_reward_func(completions=completions, answers=answers)
        self.assertEqual(rewards, [1.0, 0.0])  # Second one doesn't match exactly
        
        # Test partial matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-89</answer>"}]
        ]
        rewards = correctness_reward_func(completions=completions, answers=answers)
        self.assertEqual(rewards, [0.0, 1.0])
        
        # Test no matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-90</answer>"}]
        ]
        rewards = correctness_reward_func(completions=completions, answers=answers)
        self.assertEqual(rewards, [0.0, 0.0])
        
        # Test empty completions
        self.assertEqual(correctness_reward_func(completions=[], answers=[]), [])


if __name__ == "__main__":
    unittest.main()