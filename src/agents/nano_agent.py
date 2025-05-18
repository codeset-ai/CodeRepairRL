import os
import time
import logging
from typing import Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ThreadTimeoutError

from nano import Agent
from trl.extras.vllm_client import VLLMClient, GenerationResult

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


def _process_one(data: dict[str, Any], model: str, api_base: str, **kwargs) -> GenerationResult:
    assert (
        "repo" in data and "base_commit" in data and "problem_statement" in data
    ), "Missing required keys in input data"

    logger.info(f"[START] {data['repo']} @ {data['base_commit']}")

    try:
        temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])

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
        diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)

    finally:
        clean_repo_dir(temp_folder)

    logger.info(f"[FINISH] {data['repo']} @ {data['base_commit']}")
    return GenerationResult(
        prompt=agent.messages[:2],
        completion=agent.messages[2:],
        tools=agent.tools,
        generated_diff=diff,
    )

class MpNanoAgent(VLLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.api_base = f"http://{self.host}:{self.server_port}/v1"
        response = self.session.get(f"{self.api_base}/models").json()
        self.model = f"hosted_vllm/{response['data'][0]['id']}"

    def generate(self, data: list[dict[str, Any]], timeout: int = 300, **kwargs) -> list[GenerationResult]:
        """Deploys parallel agents to process the given data"""

        args_list = [(datum, self.model, self.api_base) for datum in data]

        results = []
        start_time = time.time()
        logger.info(f"Starting {len(data)} agent rollouts")

        ok, tout, err = 0, 0, 0
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(len(data), ctx.cpu_count())) as pool:
            futures = [pool.apply_async(_process_one, args=a, kwds=kwargs) for a in args_list]

            for fut in futures:
                try:
                    results.append(fut.get(timeout=timeout))
                    ok += 1
                except ctx.TimeoutError:
                    results.append(GenerationResult(prompt=[], completion=[], tools=[], generated_diff=""))
                    tout += 1
                except Exception as e:
                    results.append(GenerationResult(prompt=[], completion=[], tools=[], generated_diff=""))
                    err += 1

        logger.info(f"Finished rollouts {len(data)} in {time.time() - start_time:.2f}s")
        logger.info(f"Success: {ok}, Timeout: {tout}, Error: {err}")
        return results
    
class ThreadedNanoAgent(VLLMClient):
    """
    Drop-in replacement for NanoAgent that parallelises with threads instead of processes.
    Suitable when compute is I/O-bound (LLM calls) and the agent itself is lightweight.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.api_base = f"http://{self.host}:{self.server_port}/v1"
        response = self.session.get(f"{self.api_base}/models").json()
        self.model = f"hosted_vllm/{response['data'][0]['id']}"

    def generate(self, data: list[dict[str, Any]], timeout: int = 300, **kwargs) -> list[GenerationResult]:
        args_list = [(datum, self.model, self.api_base) for datum in data]

        logger.info(f"Starting {len(data)} threaded agent rollouts")
        start_time = time.time()

        results = []
        ok = tout = err = 0

        with ThreadPoolExecutor(max_workers=min(len(data), os.cpu_count() * 5)) as executor:
            futures = [executor.submit(_process_one, *a, **kwargs) for a in args_list]

            for fut in futures:
                try:
                    results.append(fut.result(timeout=timeout))
                    ok += 1
                except ThreadTimeoutError:
                    results.append(GenerationResult(prompt=[], completion=[], tools=[], generated_diff=""))
                    tout += 1
                except Exception:
                    results.append(GenerationResult(prompt=[], completion=[], tools=[], generated_diff=""))
                    err += 1

        logger.info(f"Finished rollouts {len(data)} in {time.time() - start_time:.2f}s")
        logger.info(f"Success: {ok}, Timeout: {tout}, Error: {err}")
        return results

if __name__ == "__main__":
    import time
    from src.data.swe_gym import get_swe_gym_repo_repair_dataset
    from matplotlib import pyplot as plt

    # Test different dataset sizes
    sizes = [2, 4, 8, 16]
    runs = 2
    
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42)
    
    mp_times = {size: [] for size in sizes}
    threaded_times = {size: [] for size in sizes}
    
    # Run benchmarks for each size
    for size in sizes:
        print(f"Running size {size}")
        subset = data.select(range(size))
        
        # Multiprocess runs
        for _ in range(runs):
            client = MpNanoAgent(connection_timeout=240)
            start_time = time.time()
            results = client.generate(subset, timeout=120, max_tool_calls=5, thinking=False, verbose=False)
            client.reset_prefix_cache()
            mp_times[size].append(time.time() - start_time)
            
        # Threaded runs    
        for _ in range(runs):
            client = ThreadedNanoAgent(connection_timeout=240)
            start_time = time.time()
            results = client.generate(subset, timeout=120, max_tool_calls=5, thinking=False, verbose=False)
            client.reset_prefix_cache()
            threaded_times[size].append(time.time() - start_time)
            
        # Print averages for this size
        mp_avg = sum(mp_times[size]) / runs
        threaded_avg = sum(threaded_times[size]) / runs
        print(f"\nDataset size {size}:")
        print(f"Average multiprocess time ({runs} runs): {mp_avg:.2f}s")
        print(f"Average threaded time ({runs} runs): {threaded_avg:.2f}s")

    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Calculate averages for plotting
    mp_avgs = [sum(mp_times[size])/runs for size in sizes]
    threaded_avgs = [sum(threaded_times[size])/runs for size in sizes]
    
    plt.plot(sizes, mp_avgs, 'bo-', label='Multiprocess', linewidth=2)
    plt.plot(sizes, threaded_avgs, 'ro-', label='Threaded', linewidth=2)
    
    plt.xlabel('Number of Samples')
    plt.ylabel('Average Time (seconds)')
    plt.title('Scaling Comparison: Multiprocess vs Threaded')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add value labels on points
    for i, size in enumerate(sizes):
        plt.annotate(f'{mp_avgs[i]:.1f}s', (size, mp_avgs[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{threaded_avgs[i]:.1f}s', (size, threaded_avgs[i]), textcoords="offset points", xytext=(0,-15), ha='center')
    
    plt.tight_layout()
    plt.savefig('agent_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()