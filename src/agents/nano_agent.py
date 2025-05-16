import os
import logging
import multiprocessing as mp
from typing import Any

from nano import Agent
from trl.extras.vllm_client import VLLMClient, GenerationResult

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)

mp.set_start_method("spawn")

class NanoAgent(VLLMClient):
    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        super().__init__(vllm_url)

        self.api_base = f"http://{self.host}:{self.server_port}/v1"
        response = self.session.get(f"{self.api_base}/models").json()
        self.model = f"hosted_vllm/{response['data'][0]['id'].lower()}"

    def _process_one(self, data: dict[str, Any], **kwargs) -> GenerationResult:
        """Process a single prompt and return the diff."""
        assert ("repo" in data and "base_commit" in data and "problem_statement" in data), \
            "HF dataset should be SWE-Gym esque, including: repo, base_commit and problem_statement"
        
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
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 0),
            )
            diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)
                
            logger.info(f"Agent finished for {data['repo']} at {data['base_commit']}")
            
        finally:
            logger.info(f"Cleaning up repo dir {temp_folder}")
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)
            logger.info(f"Cleanup done for {data['repo']} at {data['base_commit']}")
            
        logger.info(f"[END] Processing {data['repo']} at {data['base_commit']}")

        return GenerationResult(
            prompt=agent.messages[:2],  # System prompt and task description is our prompt
            completion=agent.messages[2:],
            tools=agent.tools,
            diff=diff,
        )

    def generate(self, data: list[dict[str, Any]], timeout: int = 300, **kwargs) -> list[GenerationResult]:
        """
        Deploys parallel agents to process the given data, returning the updated data with the generated diffs.

        Args:
            data: List of data to process
            timeout: Maximum time in seconds to wait

        Returns:
            The data with "generated_diff", "messages" and "tools" fields
        """
        with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:  # mp will "batch" the prompts if they exceed the core count
            # Start async processing of all prompts
            result = pool.map_async(self._process_one, data)
            
            try:
                generation_results = result.get(timeout=timeout)
            except mp.TimeoutError:
                logger.warning(f"Agent timeout reached after {timeout} seconds.")
                # Initialize with placeholders matching the number of original data items
                generation_results = [GenerationResult(
                    prompt=[],
                    completion=[],
                    tools=[],
                    diff="",
                )] * len(data)

        return generation_results