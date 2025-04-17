import os
import uuid
import json
import logging
import pathlib
import subprocess
import multiprocessing as mp
from typing import Dict, Any, List, Tuple

from trl.extras.vllm_client import VLLMClient   # your existing base‑class

from src.utils.git import get_head_commit_diff, handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)  # Force spawn method for multiprocessing 

class OpenHandsAgent(VLLMClient):
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        model: str = "openhands-lm-32b-v0.1",
        runtime: str = "local",          # can't use docker on berzileus
        max_iterations: int = 50,        # guard rail so the agent terminates
        timeout: int = 3 * 60,      # 2 minutes
    ):
        super().__init__(vllm_url)
        self.model = model
        self.runtime = runtime
        self.max_iters = str(max_iterations)   # CLI expects a string
        self.timeout = timeout
        # OpenHands uses a LiteLLM‑style env. Give it the local vLLM endpoint.
        os.environ.update(
            LLM_PROVIDER="openai",
            LLM_MODEL=f"openai/{model}",
            LLM_BASE_URL=vllm_url,
            LLM_API_KEY="dummy",
        )

    def _run_openhands(self, repo_dir: pathlib.Path, trajectory_file: pathlib.Path, task: str) -> int:
        env = os.environ.copy() | {
            "WORKSPACE_BASE": str(repo_dir),
            "RUNTIME": self.runtime,        # "local" ⇢ no Docker required
            "SAVE_TRAJECTORY_PATH": str(trajectory_file),
            "LOG_ALL_EVENTS": "true",       # more verbose log stream :contentReference[oaicite:0]{index=0}
            "MAX_ITERATIONS": self.max_iters,
        }
        cmd = [
            "python",
            "-m",
            "openhands.core.main",
            "-t",
            task,
        ]
        return subprocess.run(
            cmd,
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        ).returncode

    def _read_conversation(self, trajectory_file: pathlib.Path) -> List[Dict[str, str]]:
        if not trajectory_file.exists():
            logger.warning("trajectory file missing – returning empty chat")
            return []

        events = json.loads(trajectory_file.read_text())
        chat = [
            {"role": e["role"], "content": e["content"]}
            for e in events
            if e["type"] in ("user_message", "assistant_message", "system_message")
        ]
        return chat

    def _process_one(self, data: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        repo_url = handle_to_url(data["repo"])
        base = data["base_commit"]
        
        logger.info(f"Cloning {repo_url} at {base}")
        
        repo_dir = clone_repo_at_commit(repo_url, base)
        trajectory_file = repo_dir / f"traj_{uuid.uuid4().hex}.json"

        try:
            rc = self._run_openhands(repo_dir, trajectory_file, data["problem_statement"])
            if rc != 0:
                logger.error("OpenHands exited with status %s", rc)

            patch = get_head_commit_diff(repo_dir)
            chat = self._read_conversation(trajectory_file)
            return patch, chat

        finally:
            clean_repo_dir(repo_dir)

    def generate(self, data: List[Dict[str, Any]], timeout: int = 600) -> List[Dict[str, Any]]:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=min(len(data), mp.cpu_count())) as p:
            results: List[Tuple[str, List[Dict[str, str]]]] = p.map(
                self._process_one, data, chunksize=1
            )

        for job, (diff, convo) in zip(data, results):
            job["generated_diff"] = diff
            job["conversation"] = convo
        return data