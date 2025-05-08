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
                thinking=kwargs.get("thinking", False),
                max_tool_calls=kwargs.get("max_tool_calls", 20),
                temperature=kwargs.get("temperature", 0.7),
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
                collected_results = result.get(timeout=timeout)
                # Transpose the list of tuples.
                diffs, messages, tools = [list(item) for item in zip(*collected_results)]
            except mp.TimeoutError:
                logger.warning(f"Agent timeout reached after {timeout} seconds.")
                # Initialize with placeholders matching the number of original data items
                diffs = [""] * len(data)
                messages = [[]] * len(data) # list of empty lists for messages
                tools = [[]] * len(data)    # list of empty lists for tools

        # Attach the diffs to the corresponding data items
        for item, diff, msgs, tls in zip(data, diffs, messages, tools):
            item["generated_diff"] = diff
            item["messages"] = msgs
            item["tools"] = tls

        return data