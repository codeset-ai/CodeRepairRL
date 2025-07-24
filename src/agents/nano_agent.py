import os
import time
import logging
from typing import Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from nano.agent import Agent

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


@dataclass
class NanoConfig:
    model: Optional[str] = None
    api_base: str = "http://localhost:8000/v1"
    thinking: bool = False
    token_limit: int = 8192
    tool_limit: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 20
    verbose: bool = False
    remote: bool = True


def _process_one(data: dict[str, Any], config: NanoConfig) -> dict[str, Any]:
    assert "dataset" in data and "sample-id" in data and "task" in data

    logger.info(f"[START] {data['dataset']} @ {data['sample-id']}")

    agent = Agent(**asdict(config))

    try:
       agent.run(task=data["task"], sample_id=data["sample-id"])
    except Exception as e:
        logger.error(f"Error in _process_one: {type(e).__name__}: {e}")
    finally:
        token_usage = agent.token_usage
        tool_usage = agent.tool_usage
        logger.info(f"[FINISH] {data['dataset']} @ {data['sample-id']} - Tokens: {token_usage}, Tools: {tool_usage}")

    result = dict(
        prompt=agent.messages[:2],
        completion=agent.messages[2:],
        tools=agent.tools,
        **agent.tool_stats
    )
    if agent.remote:
        result["success"] = agent.is_successful()
    return result


def nano_rollout_func(data: list[dict[str, Any]], config: NanoConfig, timeout: int = 120, **kwargs) -> list[dict[str, Any]]:
    """Deploys parallel Nano agents talking to our trl vllm-serve-async endpoint to process the given data"""

    results = []
    ok, tout, err = 0, 0, 0

    logger.info(f"Starting {len(data)} agent rollouts")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(len(data), os.cpu_count())) as executor:
        futures = [executor.submit(_process_one, datum, config) for datum in data]

        for fut in as_completed(futures):
            try:
                results.append(fut.result(timeout=timeout))
                ok += 1
            except TimeoutError:
                logger.warning(f"Rollout timed out after {timeout}s")
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                tout += 1
            except Exception as e:
                logger.error(f"Rollout error: {type(e).__name__}: {e}")
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                err += 1

    logger.info(f"Finished rollouts {len(data)} in {time.time() - start_time:.2f}s")
    logger.info(f"Success: {ok}, Timeout: {tout}, Error: {err}")
    return results


if __name__ == "__main__":
    import time

    from src.data.swe_gym import get_swe_gym_repo_repair_dataset

    # Test different batch sizes for parallel timing
    batch_sizes = [2]
    runs = 1
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42)

    config = NanoConfig(model="hosted_vllm/Qwen/Qwen3-8B")

    avg_times = []

    for size in batch_sizes:
        print(f"Testing batch size {size}")
        subset = data.select(range(size))
        subset_dicts = [dict(x) for x in subset]
        times = []
        for i in range(runs):
            start_time = time.time()
            results = nano_rollout_func(subset_dicts, config, timeout=120)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")
        avg_time = sum(times) / runs
        avg_times.append(avg_time)
        print(f"Average time for batch size {size}: {avg_time:.2f}s\n")