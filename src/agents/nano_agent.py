import os
import logging
from pathlib import Path
import multiprocessing as mp
from typing import Dict, Any, List, Tuple

from nano import Agent

from trl.extras.vllm_client import AsyncVLLMClient


from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)

class NanoAgent(AsyncVLLMClient):
    """Minimal wrapper for the SubmitPatchAgent using AsyncVLLMClient."""

    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        super().__init__(vllm_url)

        self.api_base = f"http://{self.host}:{self.server_port}/v1"
        response = self.session.get(f"{self.api_base}/models").json()
        self.model = f"hosted_vllm/{response['data'][0]['id'].lower()}"

    def _process_one(self, data: Dict[str, Any], **kwargs) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]:
        """Process a single prompt and return the diff."""
        assert ("repo" in data and "base_commit" in data and "problem_statement" in data), \
            "Data should contain repo, base_commit and problem_statement"
        
        logger.info(f"[START] Processing {data['repo']} at {data['base_commit']}")
        
        original_dir = os.getcwd()
        
        try:
            logger.info(f"Cloning repo {data['repo']} at commit {data['base_commit']}")
            # Clone the repo into a temporary folder
            temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])
            logger.info(f"Cloned repo to {temp_folder}")
            
            # Change to the repo's root directory
            os.chdir(temp_folder)
            
            logger.info(f"Running agent for {data['repo']} at {data['base_commit']}")
            agent = Agent(
                model=self.model,
                api_base=self.api_base,
                thinking=kwargs.get("thinking", False),
                max_tool_calls=kwargs.get("max_tool_calls", 20),
                temperature=kwargs.get("temperature", 0.0),
            )
            
            diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)
                
            logger.info(f"Agent finished for {data['repo']} at {data['base_commit']}")
            
        finally:
            logger.info(f"Cleaning up repo dir {temp_folder}")
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)
            logger.info(f"Cleanup done for {data['repo']} at {data['base_commit']}")
            
        logger.info(f"[END] Processing {data['repo']} at {data['base_commit']}")
        return diff, agent.messages, agent.tools

    def generate(self, data: List[Dict[str, Any]], timeout: int = 300, **kwargs) -> List[Dict[str, Any]]:
        """
        Deploys parallel agents to process the given data, returning the updated data with the generated diffs.

        Args:
            data: List of data to process
            timeout: Maximum time in seconds to wait

        Returns:
            The data with "generated_diff", "messages" and "tools" fields
        """
        # Process all prompts in parallel
        with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:  # mp will "batch" the prompts if they exceed the core count
            # Start async processing of all prompts
            result = pool.map_async(self._process_one, data)
            
            try:
                # Wait for results with timeout and collect results
                diffs, messages, tools = result.get(timeout=timeout)
            except mp.TimeoutError:
                logger.warning(f"Agent timeout reached after {timeout} seconds.")
                diffs = [""] * len(data)
                messages = [[]] * len(data)
                tools = [[]] * len(data)

        # Attach the diffs to the corresponding data items
        for item, diff, msgs, tls in zip(data, diffs, messages, tools):
            item["generated_diff"] = diff
            item["messages"] = msgs
            item["tools"] = tls

        return data


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    from difflib import SequenceMatcher
    # Load example from dataset
    from src.data import get_swe_gym_repo_repair_dataset

    
    logger.info("Loading dataset")
    ds = get_swe_gym_repo_repair_dataset().shuffle(seed=43).select(range(1))
    example = dict(ds[0])
    
    logger.info(f"Testing with repo {example['repo']} at commit {example['base_commit']}")
    
    if TESTING:
        if API_PROVIDER == "openai":
            # Testing with OpenAI API
            logger.info(f"Running in TESTING mode with OpenAI API")
            
            # Check for API key before proceeding
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY environment variable is not set. Please set it when using OpenAI in TESTING mode.")
                exit(1)
        else:
            # Testing with OpenRouter API
            logger.info(f"Running in TESTING mode with OpenRouter API")
            
            # Check for API key before proceeding
            if not os.getenv("OPENROUTER_API_KEY"):
                logger.error("OPENROUTER_API_KEY environment variable is not set. Please set it when using OpenRouter in TESTING mode.")
                exit(1)
    else:
        logger.info("Running with local vLLM endpoint")
    
    # Create a temporary directory for testing
    temp_folder = clone_repo_at_commit(handle_to_url(example["repo"]), example["base_commit"])
    
    try:
        # Run SubmitPatchAgent (using either OpenAI or local vLLM based on TESTING flag)
        agent = SubmitPatchAgent(
            repo=Path(temp_folder),
            task=example["problem_statement"],
            max_tool_calls=10,
            temperature=0.0,
        )
        result = agent.run()
        
        # print(f"\nRepo: {example['repo']} at {example['base_commit']}")
        # print(f"Problem: {example['problem_statement']}")
        # print(f"Oracle diff: {example['patch']}")
        # print(f"Generated diff: {result.get('diff', '')}")

        print(f"Diff similarity: {SequenceMatcher(None, example['patch'], result.get('diff', '')).ratio()}")
        
        if result["status"] != "ok":
            logger.error(f"Agent failed: {result.get('reason', 'Unknown error')}")
        
    finally:
        # Clean up
        clean_repo_dir(temp_folder)
    
    logger.info("Test completed")


# vllm serve Qwen/Qwen3-30B-A3B \
# --host 0.0.0.0 \
# --port 8000 \
# --enable-auto-tool-choice \
# --tool-call-parser hermes \
# --enable-reasoning \
# --reasoning-parser deepseek_r1  
# --max-model-len 32768 \