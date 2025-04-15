import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.agents.aider_agent import MultiProcessAider
from src.data import get_swe_gym_repo_repair_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting test")
    logger.info("Loading dataset")
    ds = get_swe_gym_repo_repair_dataset().shuffle(seed=42).select(range(2))
    ds = [dict(x) for x in ds]  # Convert to list of dicts
    # print(dict(ds[0]))
    
    logger.info("Deploying agents")
    aider = MultiProcessAider(vllm_url="http://localhost:8000/v1")
    ds = aider.deploy(ds)
    
    logger.info(f"Processed {len(ds)} examples")
    for d in ds:
        print(d["generated_diff"])