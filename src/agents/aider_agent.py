import os
import logging
import multiprocessing as mp
from typing import Dict, Any, List
from contextlib import redirect_stdout, redirect_stderr

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from trl.extras.agent_manager import AgentManager

from src.utils.git import get_head_commit_diff, handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)  # Force spawn method for multiprocessing 


class MultiProcessAider(AgentManager):
    """Agent manager that uses multiple processes to parallelize agent deployments."""

    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        """
        Initialize the agent manager.

        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
        """
        super().__init__(vllm_url)

    def process_one(self, data: Dict[str, Any]) -> str:
        """Process a single prompt and return the diff"""

        assert ("repo" in data and "base_commit" in data and "problem_statement" in data), "Data should contain repo, base_commit and problem_statement"
        
        logger.info(f"[START] Processing {data['repo']} at {data['base_commit']}")

        original_dir = os.getcwd()

        try:
            logger.info(f"Cloning repo {data['repo']} at commit {data['base_commit']}")
            # Clone the repo into a temporary folder
            temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])
            logger.info(f"Cloned repo to {temp_folder}")

            # Change to the repo's root directory so Aider can compute the repo-map
            os.chdir(temp_folder)

            # Redirect Aider's terminal output to the void
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                logger.info(f"Running coder for {data['repo']} at {data['base_commit']}")
                # "openai" to spoof the OPENAI_API_BASE, 
                coder = Coder.create(
                    # main_model=Model("openai/Qwen/Qwen2.5-Coder-32B-Instruct"),
                    main_model=Model("haiku"),
                    io=InputOutput(yes=True),
                    suggest_shell_commands=False,
                )
                coder.run(data["problem_statement"])
                logger.info(f"Coder finished for {data['repo']} at {data['base_commit']}")
                diff = get_head_commit_diff(temp_folder)

        finally:
            logger.info(f"Cleaning up repo dir {temp_folder}")
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)
            logger.info(f"Cleanup done for {data['repo']} at {data['base_commit']}")

        logger.info(f"[END] Processing {data['repo']} at {data['base_commit']}")
        return diff

    def deploy(self, data: List[Dict[str, Any]], timeout: int = 300) -> List[Dict[str, Any]]:
        """
        Deploy parallel agents to process the given prompts, returning histories.

        Args:
            prompts: List of prompts to process
            timeout: Maximum time in seconds to wait for all prompts to complete

        Returns:
            The data with an extra "generated_diff" field
        """
        # Process all prompts in parallel
        with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:  # mp will "batch" the prompts if they exceed the core count
            # Start async processing of all prompts
            result = pool.map_async(self.process_one, data)

            try:
                # Wait for results with timeout and collect diffs
                diffs = result.get(timeout=timeout)
            except mp.TimeoutError:
                logger.warning(f"Agent timeout reached after {timeout} seconds.")
                diffs = [""] * len(data)

        # Attach the diffs to the corresponding data items
        for item, diff in zip(data, diffs):
            item["generated_diff"] = diff

        return data


class ApptainerAider(AgentManager):
    """Agent manager that uses apptainer containers to parallelize agent deployments."""

    def __init__(self, vllm_url: str = "http://localhost:8000", apptainer_image: str = "aider.sif"):
        """
        Initialize the agent manager.

        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
            apptainer_image: Path to the apptainer image
        """
        super().__init__(vllm_url)

        self.apptainer_image = apptainer_image

    def deploy(self, data: List[Dict[str, Any]], timeout: int = 300) -> List[Dict[str, Any]]:
        pass