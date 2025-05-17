import os
import time
import logging
from typing import Any
import multiprocessing as mp

from nano import Agent
from trl.extras.vllm_client import VLLMClient, GenerationResult

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


def _process_one(data: dict[str, Any], model: str, api_base: str, **kwargs) -> GenerationResult:
    assert (
        "repo" in data and "base_commit" in data and "problem_statement" in data
    ), "Missing required keys in input data"

    logger.info(f"[START] {data['repo']} @ {data['base_commit']}")
    original_dir = os.getcwd()

    try:
        temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])
        os.chdir(temp_folder)

        agent = Agent(
            model=model,
            api_base=api_base,
            thinking=kwargs.get("thinking", False),
            max_tool_calls=kwargs.get("max_tool_calls", 20),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.8),
            top_k=kwargs.get("top_k", 20),
            verbose=kwargs.get("verbose", False)
        )
        diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)

    finally:
        clean_repo_dir(temp_folder)
        os.chdir(original_dir)

    logger.info(f"[FINISH] {data['repo']} @ {data['base_commit']}")
    return GenerationResult(
        prompt=agent.messages[:2],
        completion=agent.messages[2:],
        tools=agent.tools,
        generated_diff=diff,
    )

class NanoAgent(VLLMClient):
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
        ctx = mp.get_context("fork")  # spawn launches a new, clean process every instance, so it needs to all the imports again.
        with ctx.Pool(processes=min(len(data), ctx.cpu_count())) as pool:
            futures = [pool.apply_async(_process_one, args=a) for a in args_list]

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

if __name__ == "__main__":
    from src.data.swe_gym import get_swe_gym_repo_repair_dataset

    client = NanoAgent(connection_timeout=240)
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42).select(range(4))

    results = client.generate(data, timeout=120, max_tool_calls=30, thinking=True, verbose=True)
    print("-" * 100)
    for result in results:
        print(result["generated_diff"] or "No diff produced")
        print("-" * 100)