import logging
import multiprocessing as mp
from typing import Dict, Any, List, Tuple

from trl.extras.vllm_client import AsyncVLLMClient


logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)  # Force spawn method for multiprocessing

# Not meant to be good, just a simple agent for debugging
class SimpleAgent(AsyncVLLMClient):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)

    def process_one(self, history: list[dict], **kwargs) -> list[dict]:
        url = f"http://{self.host}:{self.server_port}/v1/chat/completions"
        response = self.session.post(url, json={"messages": history})
        history.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})
        return history


    def generate(self, data: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """Code, problem statement -> first pass fix -> "No this has obvious issues, try again" -> second pass fix"""

        histories = [x["prompt"] for x in data]
        with mp.Pool(mp.cpu_count()) as pool:
            histories = pool.map(self.process_one, histories)

        histories = [history + [{"role": "user", "content": "No this has obvious issues, try again"}] for history in histories]
        with mp.Pool(mp.cpu_count()) as pool:
            histories = pool.map(self.process_one, histories)

        return [{
            "history": history,
        } for history in histories]