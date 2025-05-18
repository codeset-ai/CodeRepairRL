import time
import logging
import multiprocessing as mp
from typing import Any
import requests

logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)

def _simple_process_one(prompt: str, model: str, api_base: str, **kwargs) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # First request: ask for explanation
    messages.append({"role": "user", "content": "Explain the problem."})
    explain_response = requests.post(
        f"{api_base}/chat/completions",
        json={"model": model, "messages": messages}
    )
    explain_content = explain_response.json()["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": explain_content})
    
    # Second request: ask for solution
    messages.append({"role": "user", "content": "Now solve the problem."})
    solve_response = requests.post(
        f"{api_base}/chat/completions",
        json={"model": model, "messages": messages}
    )
    solve_content = solve_response.json()["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": solve_content})
    
    return {
        "prompt": messages[:2],
        "completion": messages[2:],
    }

def simple_rollout_func(data: list[dict[str, Any]], timeout: int = 60, **kwargs) -> list[dict[str, Any]]:
    api_base = "http://localhost:8000/v1"
    model = requests.get(f"{api_base}/models").json()["data"][0]["id"]
    
    results = []
    logger.info(f"Starting {len(data)} simple agent rollouts")
    start_time = time.time()
    with mp.Pool(processes=min(len(data), mp.cpu_count())) as pool:
        futures = [pool.apply_async(_simple_process_one, args=(datum["prompt"], model, api_base), kwds=kwargs) for datum in data]
        for fut in futures:
            try:
                results.append(fut.get(timeout=timeout))
            except mp.TimeoutError:
                results.append({"prompt": [], "completion": []})
            except Exception:
                results.append({"prompt": [], "completion": []})
    logger.info(f"Finished rollouts {len(data)} in {time.time() - start_time:.2f}s")
    return results

        