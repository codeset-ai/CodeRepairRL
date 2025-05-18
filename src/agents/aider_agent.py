import time
import os
import logging
import multiprocessing as mp
from typing import Any
from contextlib import redirect_stdout, redirect_stderr

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from src.utils.git import get_head_commit_diff, handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)

def _aider_process_one(data: dict[str, Any], **kwargs) -> dict:
    assert "repo" in data and "base_commit" in data and "problem_statement" in data
    
    logger.info(f"[START] Processing {data['repo']} at {data['base_commit']}")
    original_dir = os.getcwd()
    try:
        temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])
        os.chdir(temp_folder)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            coder = Coder.create(
                main_model=Model("haiku"),
                io=InputOutput(yes=True),
                suggest_shell_commands=False,
            )
            coder.run(data["problem_statement"])
            diff = get_head_commit_diff(temp_folder)
            messages = coder.format_chat_chunks().all_messages()
    finally:
        clean_repo_dir(temp_folder)
        os.chdir(original_dir)

    logger.info(f"[END] Processing {data['repo']} at {data['base_commit']}")
    return {
        "prompt": [{"role": "user", "content": data["problem_statement"]}],
        "completion": messages,
        "generated_diff": diff
    }

def aider_rollout_func(data: list[dict[str, Any]], timeout: int = 300, **kwargs) -> list[dict[str, Any]]:
    results = []
    with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:
        futures = [pool.apply_async(_aider_process_one, args=(datum,), kwds=kwargs) for datum in data]
        for fut in futures:
            try:
                results.append(fut.get(timeout=timeout))
            except mp.TimeoutError:
                results.append({"prompt": [], "completion": []})
            except Exception:
                results.append({"prompt": [], "completion": []})
    return results