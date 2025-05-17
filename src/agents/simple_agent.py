import logging
import multiprocessing as mp
from typing import Any

from trl.extras.vllm_client import VLLMClient


logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)  # Force spawn method for multiprocessing

# Not meant to be good, just a simple agent for debugging
class SimpleAgent(VLLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_one(self, history: list[dict], **kwargs) -> list[dict]:
        url = f"http://{self.host}:{self.server_port}/v1/chat/completions"
        response = self.session.post(url, json={"messages": history})
        history.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})
        return history


    def generate(self, data: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """"""

        histories = [x["prompt"] for x in data]
        with mp.Pool(mp.cpu_count()) as pool:
            histories = pool.map(self.process_one, histories)

        histories = [history + [{"role": "user", "content": "No this has obvious issues, try again"}] for history in histories]
        with mp.Pool(mp.cpu_count()) as pool:
            histories = pool.map(self.process_one, histories)

        