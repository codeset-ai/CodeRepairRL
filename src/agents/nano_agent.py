import time
import logging
from typing import Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        max_tool_calls=kwargs.get("max_tool_calls", 30),
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
    model = f"hosted_vllm/{model}"

    results = []
    ok, tout, err = 0, 0, 0

    logger.info(f"Starting {len(data)} agent rollouts")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(len(data), mp.cpu_count())) as executor:
        futures = [executor.submit(_process_one, datum, model, api_base, **kwargs) for datum in data]

        for fut in as_completed(futures):
            try:
                results.append(fut.result(timeout=timeout))
                ok += 1
            except TimeoutError:
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                tout += 1
            except:
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                err += 1

    logger.info(f"Finished rollouts {len(data)} in {time.time() - start_time:.2f}s")
    logger.info(f"Success: {ok}, Timeout: {tout}, Error: {err}")
    return results


if __name__ == "__main__":
    import time
    from src.data.swe_gym import get_swe_gym_repo_repair_dataset
    from matplotlib import pyplot as plt

    from src.agents.nano_agent import nano_rollout_func

    # Test different batch sizes for parallel timing
    batch_sizes = [1, 2, 4, 8, 16]
    runs = 2
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42)

    avg_times = []

    for size in batch_sizes:
        print(f"Testing batch size {size}")
        subset = data.select(range(size))
        subset_dicts = [dict(x) for x in subset]
        times = []
        for i in range(runs):
            start_time = time.time()
            results = nano_rollout_func(subset_dicts, timeout=120)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")
        avg_time = sum(times) / runs
        avg_times.append(avg_time)
        print(f"Average time for batch size {size}: {avg_time:.2f}s\n")

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, avg_times, 'bo-', linewidth=2)
    plt.xlabel('Batch Size (Parallel Samples)')
    plt.ylabel('Average Time (seconds)')
    plt.title('Nano Agent Parallel Batch Timing')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('nano_agent_parallel_time.png', dpi=300, bbox_inches='tight')
    plt.close()