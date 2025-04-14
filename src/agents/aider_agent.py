import os
import logging
import multiprocessing
from typing import Dict, Any, List
from contextlib import redirect_stdout, redirect_stderr

import requests

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from trl.extras.agent_manager import AgentManager

from src.utils.git import get_head_commit_diff, handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


class MultiProcessAider(AgentManager):
    """Agent manager that uses multiple processes to parallelize agent deployments."""

    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        """
        Initialize the agent manager.

        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
        """
        super().__init__(vllm_url)

    def process_one(self, prompt: Dict[str, Any]):
        """Process a single prompt and return a completion"""

        assert ("repo" in prompt and "base_commit" in prompt and "problem_statement" in prompt), "Data should contain repo, base_commit and problem_statement"

        original_dir = os.getcwd()

        try:
            # Clone the repo into a temporary folder
            temp_folder = clone_repo_at_commit(handle_to_url(prompt["repo"]), prompt["base_commit"])

            # Change to the repo's root directory so Aider can compute the repo-map
            os.chdir(temp_folder)

            # Redirect Aider's terminal output to the void
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                # "openai" to spoof the OPENAI_API_BASE, 
                coder = Coder.create(
                    # main_model=Model("openai/Qwen/Qwen2.5-Coder-32B-Instruct"),
                    main_model=Model("haiku"),
                    io=InputOutput(yes=True),
                    suggest_shell_commands=False,
                )
                coder.run(prompt["problem_statement"])
                diff = get_head_commit_diff(temp_folder)

        finally:
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)

        return diff

    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        """
        Deploy parallel agents to process the given prompts, returning histories.

        Args:
            prompts: List of prompts to process
            timeout: Maximum time in seconds to wait for all prompts to complete

        Returns:
            List of unordered results
        """
        # Process all prompts in parallel
        with multiprocessing.Pool() as pool:
            # Start async processing of all prompts
            result = pool.map_async(self.process_one, prompts)

            try:
                # Wait for results with timeout
                print(result.get(timeout=timeout))
            except multiprocessing.TimeoutError:
                logger.warning(f"Agent timeout reached after {timeout} seconds.")


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


    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        pass
