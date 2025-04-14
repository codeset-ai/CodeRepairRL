import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.agents.aider_agent import MultiProcessAider
from src.data import get_swe_gym_repo_repair_dataset


if __name__ == "__main__":
    ds = get_swe_gym_repo_repair_dataset().select(range(2))
    aider = MultiProcessAider(vllm_url="http://localhost:8000/v1")
    aider.deploy(ds)