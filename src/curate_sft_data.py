import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
from datasets import Dataset, DatasetInfo
from huggingface_hub import login
from huggingface_hub.utils import get_token

from src.data.swe_gym import get_swe_gym_curation_dataset
from src.agents.nano_agent import _process_one, NanoConfig
from src.rewards.diff import unified_diff_similarity_reward_func, unified_diff_file_match_reward_func, unified_diff_similarity_reward_func_test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy loggers
for noisy in ("httpx", "LiteLLM", "transformers.tokenization_utils_base"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

@dataclass
class CurationConfig:
    # Dataset configuration
    input_dataset_name: str = "SWE-Gym/SWE-Gym-Lite"
    curation_ratio: float = 0.2
    output_dataset_name: str = "ASSERT-KTH/SWE-Gym-Nano-SFT"
    dataset_version: str = "v1.0"
    push_to_hub: bool = False
    hf_token: Optional[str] = None  # HuggingFace token for pushing to hub
    
    # Rollout configuration
    num_rollouts_per_problem: int = 8
    timeout: int = 120
    max_workers: int = 4  # ThreadPoolExecutor max workers
    max_problems: Optional[int] = None  # Maximum number of problems to process (for testing)
    

@dataclass
class Config:
    curation: CurationConfig = field(default_factory=CurationConfig)
    agent: NanoConfig = field(default_factory=NanoConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_curation_config", node=Config, group="")

def process_one_with_reward(problem_data: dict[str, Any], config: NanoConfig) -> dict[str, Any]:
    """
    Helper function that wraps _process_one and calculates reward.
    """
    result = _process_one(problem_data, config)
    
    # Calculate rewards using same approach as train_grpo
    generated_diff = result["generated_diff"]
    
    # Calculate individual rewards (keeping patch and test_patch separate)
    file_match_reward = unified_diff_file_match_reward_func([problem_data["patch"]], [generated_diff])[0]
    similarity_reward = unified_diff_similarity_reward_func([problem_data["patch"]], [generated_diff])[0]
    test_similarity_reward = unified_diff_similarity_reward_func_test([problem_data["test_patch"]], [generated_diff])[0]
    
    # Combine rewards with same weights as train_grpo (0.2 file_match + 0.4 similarity + 0.4 test_similarity)
    combined_reward = 0.2 * file_match_reward + 0.4 * similarity_reward + 0.4 * test_similarity_reward
    
    # Add all rewards to result
    result["file_match_reward"] = file_match_reward
    result["similarity_reward"] = similarity_reward
    result["test_similarity_reward"] = test_similarity_reward
    result["reward"] = combined_reward
    result["instance_id"] = problem_data["instance_id"]
    result["problem_statement"] = problem_data["problem_statement"]
    result["repo"] = problem_data["repo"]
    result["base_commit"] = problem_data["base_commit"]
    result["oracle_diff"] = problem_data["patch"]
    result["oracle_test_diff"] = problem_data["test_patch"]
    
    return result



@hydra.main(version_base="1.1", config_path="conf", config_name="curation_config")
def main(cfg: Config) -> None:
    # Login to HuggingFace if pushing to hub
    if cfg.curation.push_to_hub:
        # Check if already logged in
        existing_token = get_token()
        if existing_token:
            logger.info("Already authenticated with HuggingFace Hub")
        elif cfg.curation.hf_token:
            login(token=cfg.curation.hf_token)
            logger.info("Logged in to HuggingFace Hub with provided token")
        else:
            login()  # Will prompt for token
            logger.info("Logged in to HuggingFace Hub")
    
    # Load SWE-Gym dataset
    logger.info("Loading SWE-Gym dataset...")
    dataset = get_swe_gym_curation_dataset(cfg.curation.input_dataset_name, cfg.curation.curation_ratio)
    
    # Limit dataset size for testing
    if cfg.curation.max_problems:
        dataset = dataset.select(range(min(cfg.curation.max_problems, len(dataset))))
        logger.info(f"Limited to {len(dataset)} problems for testing")
    
    logger.info(f"Processing {len(dataset)} problems from SWE-Gym-Lite")
    
    # Create all rollout tasks upfront
    all_rollout_tasks = []
    for problem_data in dataset:
        for _ in range(cfg.curation.num_rollouts_per_problem):
            all_rollout_tasks.append(dict(problem_data))
    
    logger.info(f"Created {len(all_rollout_tasks)} total rollout tasks")
    
    # Process all rollouts using single ThreadPoolExecutor
    all_solutions = []
    
    with ThreadPoolExecutor(max_workers=cfg.curation.max_workers) as executor:
        futures = [executor.submit(process_one_with_reward, task, cfg.agent) for task in all_rollout_tasks]
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=cfg.curation.timeout)
                solution = {
                    "instance_id": result["instance_id"],
                    "problem_statement": result["problem_statement"],
                    "repo": result["repo"],
                    "base_commit": result["base_commit"],
                    "oracle_diff": result["oracle_diff"],
                    "oracle_test_diff": result["oracle_test_diff"],
                    "generated_diff": result["generated_diff"],
                    "reward": result["reward"],
                    "file_match_reward": result["file_match_reward"],
                    "similarity_reward": result["similarity_reward"],
                    "test_similarity_reward": result["test_similarity_reward"],
                    "messages": result["prompt"] + result["completion"],
                    "tools": result["tools"]
                }
                if solution["generated_diff"] == "":
                    logger.warning(f"Generated diff is empty for problem {solution['instance_id']}")
                    continue
                all_solutions.append(solution)
            except Exception as e:
                logger.error(f"Rollout failed: {e}")
                # Skip failed rollouts
                continue
    
    logger.info(f"Curation completed: {len(all_solutions)} solutions from {len(dataset)} problems")
    
    if not all_solutions:
        logger.warning("No solutions found! Check your model setup.")
        return
    
    # Create HuggingFace dataset
    logger.info("Creating HuggingFace dataset...")
    
    # Create dataset info with detailed description like in stack.py
    info = DatasetInfo(
        description=f"""SFT data generated by Nano, a terminal-based coding agent that uses tools like shell 
to navigate repositories and solve software engineering problems from the SWE-Gym dataset.

## Dataset Structure
- `instance_id`: Problem identifier from SWE-Gym
- `problem_statement`: Description of the coding problem
- `repo`: Repository information
- `base_commit`: Base commit hash
- `oracle_diff`: Ground truth patch diff
- `oracle_test_diff`: Ground truth test diff
- `generated_diff`: Agent-generated solution diff
- `reward`: Combined weighted reward score (0.0 to 1.0)
- `file_match_reward`: File matching reward (0.0 to 1.0, weight: 0.2)
- `similarity_reward`: Patch similarity reward (0.0 to 1.0, weight: 0.4)
- `test_similarity_reward`: Test similarity reward (0.0 to 1.0, weight: 0.4)
- `messages`: Agent conversation with problem and solution
- `tools`: Shell and navigation tools used by the agent

## Generation Process
- {cfg.curation.num_rollouts_per_problem} rollouts per problem using Nano agent
- All solutions included with reward scores for post-processing filtering
- Generated with temperature {cfg.agent.temperature}, top-p {cfg.agent.top_p}
        """
    )
    
    curated_dataset = Dataset.from_list(all_solutions, info=info)
    
    # Save dataset locally first
    local_path = f"data/{cfg.curation.output_dataset_name.replace('/', '-')}-{cfg.curation.dataset_version}"
    curated_dataset.save_to_disk(local_path)
    logger.info(f"Dataset saved locally to {local_path}")
    
    # Push to HuggingFace Hub if requested
    if cfg.curation.push_to_hub:
        logger.info(f"Pushing dataset to HuggingFace Hub: {cfg.curation.output_dataset_name}")
        try:
            curated_dataset.push_to_hub(
                cfg.curation.output_dataset_name,
                commit_message=f"Curated Nano SFT data v{cfg.curation.dataset_version} with {len(all_solutions)} solutions"
            )
            logger.info("Successfully pushed dataset to HuggingFace Hub")
            logger.info(f"Dataset URL: https://huggingface.co/datasets/{cfg.curation.output_dataset_name}")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hub: {e}")
            logger.info(f"Dataset is still available locally at {local_path}")
    
    logger.info("Data curation completed successfully!")


if __name__ == "__main__":
    main()