import time
import logging
import requests
from typing import Any
import multiprocessing as mp

from src.utils.git import get_head_commit_diff, handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)

def _codex_process_one(data: dict[str, Any], model: str, api_base: str, **kwargs) -> dict[str, Any]:
    logger.info(f"[START] {data['repo']} @ {data['base_commit'][:7]}")

    # TODO:
    logger.info(f"[FINISH] {data['repo']} @ {data['base_commit'][:7]}")
    return {
        "prompt": [],
        "completion": [],
    }

def codex_rollout_func(data: list[dict[str, Any]], timeout: int = 600, **kwargs) -> list[dict[str, Any]]:
    api_base = "http://localhost:8000/v1"
    model = requests.get(f"{api_base}/models").json()["data"][0]["id"]

    results = []
    ok, tout, err = 0, 0, 0
    logger.info(f"Starting {len(data)} codex agent rollouts")
    start_time = time.time()
    with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:
        futures = [pool.apply_async(_codex_process_one, args=(datum, model, api_base), kwds=kwargs) for datum in data]
        for fut in futures:
            try:
                results.append(fut.get(timeout=timeout))
                ok += 1
            except mp.TimeoutError:
                results.append({"prompt": [], "completion": []})
                tout += 1
            except Exception:
                results.append({"prompt": [], "completion": []})
                err += 1
    logger.info(f"Finished codex rollouts {len(data)} in {time.time() - start_time:.2f}s")
    logger.info(f"Success: {ok}, Timeout: {tout}, Error: {err}")
    return results