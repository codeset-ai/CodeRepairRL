import os
import json
from pathlib import Path
import uuid
import subprocess
import multiprocessing as mp
import logging
from typing import Dict, Any, List, Tuple
from contextlib import redirect_stdout, redirect_stderr
# from trl.extras.vllm_client import VLLMClient

from src.utils.git import (
    get_head_commit_diff,
    handle_to_url,
    clone_repo_at_commit,
    clean_repo_dir,
)

logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)

class SweAgent:
    """Parallel runner for SWE‑Agent via its CLI."""

    def __init__(
        self,
        # vllm_url: str = "http://localhost:8000/v1",
        config: str = "swe_agent_haiku.yaml",
        timeout: int = 3 * 60,
    ):
        self.config = config
        self.timeout = timeout
        # no extra env needed for SWE‑Agent beyond API_BASE/API_KEY if using vLLM

    def _process_one(self, data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        assert all(k in data for k in ("repo", "base_commit", "problem_statement")), \
            "Data must include repo, base_commit and problem_statement"
        
        repo_url = handle_to_url(data["repo"])
        base = data["base_commit"]
        logger.info(f"[START] SWE‑Agent on {repo_url}@{base}")

        # 1) clone & checkout
        repo_dir = clone_repo_at_commit(repo_url, base)
        repo_dir = Path(repo_dir)
        traj_path = repo_dir / f"traj_{uuid.uuid4().hex}.json"

        try:
            # 2) run sweagent CLI
            cmd = [
                "sweagent", "run",
                "--config", self.config,
                "--problem_statement.text", data["problem_statement"],
                "--env.repo.path", str(repo_dir),
                "--env.repo.type", "local",
                "--agent.recorder.save_path", str(traj_path),
            ]
            # optional: silence stdout/stderr from the subprocess
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                subprocess.run(
                    cmd,
                    cwd=repo_dir,
                    env=os.environ.copy(),
                    stdout=devnull,
                    stderr=devnull,
                    timeout=self.timeout,
                    check=False,
                )

            # 3) collect results
            diff = get_head_commit_diff(repo_dir)
            if traj_path.exists():
                chat = json.loads(traj_path.read_text())
            else:
                logger.warning("No trajectory file, returning empty chat")
                chat = []

        finally:
            # 4) cleanup
            clean_repo_dir(str(repo_dir))
            logger.info(f"[END] SWE‑Agent on {repo_url}@{base}")

        return diff, chat

    def generate(
        self,
        data: List[Dict[str, Any]],
        timeout: int = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        data: list of { repo, base_commit, problem_statement }
        returns each item extended with .generated_diff and .conversation
        """
        timeout = timeout or self.timeout
        with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:
            results = pool.map(self._process_one, data)

        for item, (diff, chat) in zip(data, results):
            item["generated_diff"] = diff
            item["conversation"] = chat
        return data

if __name__ == "__main__":
    agent = SweAgent(config="swe_agent_haiku.yaml")
    data = [
        {"repo": "ASSERT-KTH/trl", "base_commit": "main", "problem_statement": "What is the difference between async and sync vllm clients?"}
    ]
    results = agent.generate(data)
    # print(results)