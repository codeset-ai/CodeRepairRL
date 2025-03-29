import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.rewards import (
    diff_format_reward_func,
    diff_similarity_reward_func
)


class TestRepairRewards(unittest.TestCase):
    """Test cases for repair-based reward functions in src/utils/rewards.py."""
    
    def test_diff_format_reward_func(self):
        """Test the diff format reward function using a direct mock of the reward function."""
        # Create test completions with realistic LLM responses
        search_replace_completion = [{
            "content": (
                "<think>\n"
                "I need to fix the function name from 'hello' to 'hello_world'.\n"
                "</think>\n"
                "<answer>\n"
                "```\n"
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print(\"Hello\")\n"
                "=======\n"
                "def hello_world():\n"
                "    print(\"Hello World\")\n"
                ">>>>>>> REPLACE\n"
                "```\n"
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
            [no_diff_completion[0]]
        ]
        
        # Test for search_replace diff type
        sr_rewards = diff_format_reward_func(None, completions)
        self.assertEqual(len(sr_rewards), 3)
        self.assertGreater(sr_rewards[0], 0.5)  # Should have high quality for search/replace format
        self.assertEqual(sr_rewards[1], 0.0)    # Unified diff format should get 0 for search/replace
        self.assertEqual(sr_rewards[2], 0.0)    # No diff should get 0
        

    def test_diff_similarity_reward_func(self):
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
                "```\n"
                "<<<<<<< SEARCH\n"
                "def hello():\n"
                "    print(\"Hello\")\n"
                "=======\n"
                "def hello_world():\n"
                "    print(\"Hello World\")\n"
                ">>>>>>> REPLACE\n"
                "```\n"
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
                "```\n"
                "@@ -1,3 +1,3 @@\n"
                "-def old_function():\n"
                "+def new_function():\n"
                "     print(\"Hello\")\n"
                "```\n"
                "</answer>\n"
            )
        }]
        
        completions = [
            exact_match_completion,
            non_match_completion
        ]
        
        # Reference diffs that we'll compare against
        diff_data = [
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
        
        
        # Test for search_replace diff type
        sr_rewards = diff_similarity_reward_func(prompts, completions, diffs=diff_data)
        self.assertEqual(len(sr_rewards), 2)
        self.assertGreater(sr_rewards[0], 0.5)  # First should have high similarity
        self.assertLess(sr_rewards[1], 0.5)     # Second should have low similarity

    def test_reward_function_empty_completions(self):
        """Test reward functions with empty completions list."""
        
        empty_completions = []
        empty_prompts = []
        empty_diff = []
        
        self.assertEqual(diff_format_reward_func(empty_prompts, empty_completions), [])
        self.assertEqual(diff_similarity_reward_func(empty_prompts, empty_completions, empty_diff), [])


if __name__ == "__main__":
    unittest.main() 