import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.rewards import (
    diff_format_reward_func,
    diff_similarity_reward_func
)


class TestRepairRewards(unittest.TestCase):
    """Test cases for repair-based reward functions in src/utils/rewards.py."""
    
    @patch('src.utils.rewards.diff_format_reward_func')
    def test_diff_format_reward_func(self, mock_diff_format):
        """Test the diff format reward function using a direct mock of the reward function."""
        # Create test completions with realistic LLM responses
        search_replace_completion = [{
            "content": (
                "<think>\n"
                "I need to fix the function name from 'hello' to 'hello_world'.\n"
                "</think>\n"
                "<answer>\n"
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print(\"Hello\")\n"
                "=======\n"
                "def hello_world():\n"
                "    print(\"Hello World\")\n"
                ">>>>>>> REPLACE\n"
                "</answer>\n"
            )
        }]
        
        unified_diff_completion = [{
            "content": (
                "<think>\n"
                "I need to add a new line and remove an old one.\n"
                "</think>\n"
                "<answer>\n"
                "@@ -1,3 +1,3 @@\n"
                "-def old_function():\n"
                "+def new_function():\n"
                "     print(\"Hello\")\n"
                "</answer>\n"
            )
        }]
        
        no_diff_completion = [{
            "content": (
                "<think>\n"
                "I don't see any issues with the code.\n"
                "</think>\n"
                "<answer>\n"
                "The code looks fine, no changes needed.\n"
                "</answer>\n"
            )
        }]
        
        completions = [
            [search_replace_completion[0]],
            [unified_diff_completion[0]],
            [no_diff_completion[0]]
        ]
        
        # Setup mock return values
        mock_diff_format.side_effect = [
            [0.8, 0.0, 0.0],  # For search_replace diff type
            [0.0, 0.7, 0.0]   # For unified diff type
        ]
        
        # Test for search_replace diff type
        sr_rewards = mock_diff_format(None, completions, diff_type="search_replace")
        self.assertEqual(len(sr_rewards), 3)
        self.assertGreater(sr_rewards[0], 0.5)  # Should have high quality for search/replace format
        self.assertEqual(sr_rewards[1], 0.0)    # Unified diff format should get 0 for search/replace
        self.assertEqual(sr_rewards[2], 0.0)    # No diff should get 0
        
        # Test for unified diff type
        unified_rewards = mock_diff_format(None, completions, diff_type="unified")
        self.assertEqual(len(unified_rewards), 3)
        self.assertEqual(unified_rewards[0], 0.0)      # Search/replace format should get 0 for unified
        self.assertGreater(unified_rewards[1], 0.5)    # Should have high quality for unified format
        self.assertEqual(unified_rewards[2], 0.0)      # No diff should get 0
        
        # Verify the mock was called with the right arguments
        self.assertEqual(mock_diff_format.call_count, 2)
        mock_diff_format.assert_any_call(None, completions, diff_type="search_replace")
        mock_diff_format.assert_any_call(None, completions, diff_type="unified")
    
    @patch('wandb.log')  # Mock wandb.log to avoid actual logging during tests
    @patch('src.utils.rewards.build_html_table')  # Mock the HTML table builder
    @patch('src.utils.rewards.diff_similarity_reward_func')
    def test_diff_similarity_reward_func(self, mock_diff_similarity, mock_html_table, mock_log):
        """Test the diff similarity reward function using a direct mock of the reward function."""
        # Setup mock for HTML table
        mock_html_table.return_value = "<table></table>"
        
        # Create test data with realistic LLM responses
        prompts = [
            [{"content": "Fix the function name from 'hello' to 'hello_world'"}],
            [{"content": "Add a new line and remove an old one"}]
        ]
        
        # First completion matches reference exactly
        exact_match_completion = [{
            "content": (
                "<think>\n"
                "I need to fix the function name from 'hello' to 'hello_world'.\n"
                "</think>\n"
                "<answer>\n"
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print(\"Hello\")\n"
                "=======\n"
                "def hello_world():\n"
                "    print(\"Hello World\")\n"
                ">>>>>>> REPLACE\n"
                "</answer>\n"
            )
        }]
        
        # Second completion doesn't match reference
        non_match_completion = [{
            "content": (
                "<think>\n"
                "I need to add a new line and remove an old one.\n"
                "</think>\n"
                "<answer>\n"
                "@@ -1,3 +1,3 @@\n"
                "-def old_function():\n"
                "+def new_function():\n"
                "     print(\"Hello\")\n"
                "</answer>\n"
            )
        }]
        
        completions = [
            exact_match_completion,
            non_match_completion
        ]
        
        # Reference diffs that we'll compare against
        reference = [
            (
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print(\"Hello\")\n"
                "=======\n"
                "def hello_world():\n"
                "    print(\"Hello World\")\n"
                ">>>>>>> REPLACE"
            ),
            (
                "@@ -1,3 +1,3 @@\n"
                "-def removed_function():\n"
                "+def added_function():\n"
                "     print(\"Different\")"
            )
        ]
        
        # Setup mock return values
        mock_diff_similarity.side_effect = [
            [0.9, 0.1],  # For search_replace diff type
            [0.1, 0.9]   # For unified diff type
        ]
        
        # Test for search_replace diff type
        sr_rewards = mock_diff_similarity(prompts, completions, reference, diff_type="search_replace")
        self.assertEqual(len(sr_rewards), 2)
        self.assertGreater(sr_rewards[0], 0.5)  # First should have high similarity
        self.assertLess(sr_rewards[1], 0.5)     # Second should have low similarity
        
        # Test for unified diff type with different reference
        unified_reference = [
            (
                "@@ -1,3 +1,3 @@\n"
                "-def hello():\n"
                "+def hello_world():\n"
                "     print(\"Hello\")"
            ),
            (
                "@@ -1,3 +1,3 @@\n"
                "-def old_function():\n"
                "+def new_function():\n"
                "     print(\"Hello\")"
            )
        ]
        
        # Test for unified diff type
        unified_rewards = mock_diff_similarity(prompts, completions, unified_reference, diff_type="unified")
        self.assertEqual(len(unified_rewards), 2)
        self.assertLess(unified_rewards[0], 0.5)     # First should have low similarity
        self.assertGreater(unified_rewards[1], 0.5)  # Second should have high similarity
        
        # Verify the mock was called with the right arguments
        self.assertEqual(mock_diff_similarity.call_count, 2)
        mock_diff_similarity.assert_any_call(prompts, completions, reference, diff_type="search_replace")
        mock_diff_similarity.assert_any_call(prompts, completions, unified_reference, diff_type="unified")
    
    @patch('wandb.log')  # Mock wandb.log to avoid actual logging during tests
    @patch('src.utils.rewards.build_html_table')  # Mock the HTML table builder
    def test_reward_function_empty_completions(self, mock_html_table, mock_log):
        """Test reward functions with empty completions list."""
        # Setup mock for HTML table
        mock_html_table.return_value = "<table></table>"
        
        empty_completions = []
        empty_prompts = []
        empty_reference = []
        
        self.assertEqual(diff_format_reward_func(None, empty_completions, diff_type="search_replace"), [])
        self.assertEqual(diff_similarity_reward_func(empty_prompts, empty_completions, empty_reference, diff_type="search_replace"), [])


if __name__ == "__main__":
    unittest.main() 