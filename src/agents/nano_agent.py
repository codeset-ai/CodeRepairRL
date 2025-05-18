import time
import logging
from typing import Any
import multiprocessing as mp

import requests
from nano import Agent

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


def _process_one(data: dict[str, Any], model: str, api_base: str, **kwargs) -> dict[str, Any]:
    assert "repo" in data and "base_commit" in data and "problem_statement" in data

    logger.info(f"[START] {data['repo']} @ {data['base_commit'][:7]}")

    agent = Agent(
        model=model,
        api_base=api_base,
        thinking=kwargs.get("thinking", False),
        max_tool_calls=kwargs.get("max_tool_calls", 5),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.8),
        top_k=kwargs.get("top_k", 20),
        verbose=kwargs.get("verbose", False)
    )

    try:
        temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])
        diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)
    except:
        return dict(
            prompt=agent.messages[:2],
            completion=agent.messages[2:],
            tools=agent.tools,
            generated_diff="",
        ) # Grab context window errors here and return partial results

    finally:
        clean_repo_dir(temp_folder)
        logger.info(f"[FINISH] {data['repo']} @ {data['base_commit'][:7]}")

    return dict(
        prompt=agent.messages[:2],
        completion=agent.messages[2:],
        tools=agent.tools,
        generated_diff=diff,
    )


def nano_rollout_func(data: list[dict[str, Any]], timeout: int = 300, **kwargs) -> list[dict[str, Any]]:
    """Deploys parallel Nano agents talking to our trl vllm-serve-async endpoint to process the given data"""

    api_base = "http://localhost:8000/v1"
    model = requests.get(f"{api_base}/models").json()["data"][0]["id"]

    results = []
    ok, tout, err = 0, 0, 0

    logger.info(f"Starting {len(data)} agent rollouts")
    start_time = time.time()
    with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:
        futures = [pool.apply_async(_process_one, args=(datum, model, api_base), kwds=kwargs) for datum in data]

        for fut in futures:
            try:
                results.append(fut.get(timeout=timeout))
                ok += 1
            except mp.TimeoutError:
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                tout += 1
            except:
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                err += 1

    logger.info(f"Finished rollouts {len(data)} in {time.time() - start_time:.2f}s")
    logger.info(f"Success: {ok}, Timeout: {tout}, Error: {err}")
    return results

