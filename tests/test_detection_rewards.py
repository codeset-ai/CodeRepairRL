import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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

    @patch('wandb.log')  # Mock wandb.log to avoid actual logging during tests
    @patch('src.utils.rewards.build_html_table')  # Mock the HTML table builder
    def test_correctness_reward_func(self, mock_html_table, mock_log):
        """Test the correctness reward function."""
        # Setup mock for HTML table
        mock_html_table.return_value = "<table></table>"
        
        # Setup test data
        prompts = [[{"content": "prompt1"}], [{"content": "prompt2"}]]
        completions = [
            [{"content": "<answer>CWE-79</answer>"}],
            [{"content": "<answer>CWE-89: SQL Injection</answer>"}]
        ]
        answers = ["CWE-79", "CWE-89"]
        
        # Test exact matches
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [1.0, 0.0])  # Second one doesn't match exactly
        
        # Test partial matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-89</answer>"}]
        ]
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [0.0, 1.0])
        
        # Test no matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-90</answer>"}]
        ]
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [0.0, 0.0])
        
        # Test empty completions
        empty_completions = []
        self.assertEqual(correctness_reward_func([], empty_completions, []), [])


if __name__ == "__main__":
    unittest.main()