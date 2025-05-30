import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
from datasets import Dataset, DatasetInfo
from huggingface_hub import login
from huggingface_hub.utils import get_token

from src.data.swe_gym import get_swe_gym_curation_dataset
from src.agents.nano_agent import nano_rollout_func, NanoConfig
from src.rewards.diff import unified_diff_similarity_reward_func

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
    max_problems: Optional[int] = None  # Maximum number of problems to process (for testing)
    
    # Filtering configuration
    reward_threshold: float = 0.4
    min_solutions_per_problem: int = 1
    max_solutions_per_problem: int = 3

@dataclass
class Config:
    curation: CurationConfig = field(default_factory=CurationConfig)
    agent: NanoConfig = field(default_factory=NanoConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_curation_config", node=Config, group="")

def curate_problem(problem_data: dict[str, Any], config: Config) -> list[dict]:
    """
    Run multiple rollouts for a single problem and filter high-quality solutions.
    
    Args:
        problem_data: Single problem from SWE-Gym dataset
        config: Curation configuration
        
    Returns:
        List of high-quality solutions with rewards > threshold
    """
    # Create multiple copies of the problem for parallel rollouts
    rollout_data = [dict(problem_data) for _ in range(config.curation.num_rollouts_per_problem)]
    
    logger.info(f"Running {config.curation.num_rollouts_per_problem} rollouts for {problem_data['instance_id']}")
    
    # Run parallel rollouts
    results = nano_rollout_func(
        rollout_data,
        config=config.agent,
        timeout=config.curation.timeout
    )
    
    # Extract generated diffs
    generated_diffs = [result["generated_diff"] for result in results]
    

    patch_list = [problem_data["patch"]] * len(generated_diffs)
    rewards = unified_diff_similarity_reward_func(patch_list, generated_diffs)

    # Filter solutions above threshold
    high_quality_solutions = []
    for result, reward in zip(results, rewards):
        if reward >= config.curation.reward_threshold and result.get("generated_diff"):
            solution = {
                "instance_id": problem_data["instance_id"],
                "problem_statement": problem_data["problem_statement"],
                "repo": problem_data["repo"],
                "base_commit": problem_data["base_commit"],
                "oracle_diff": problem_data["patch"],
                "generated_diff": result["generated_diff"],
                "reward": reward,
                "messages": result["prompt"] + result["completion"],
                "tools": result["tools"]
            }
            high_quality_solutions.append(solution)
    
    # Sort by reward (descending) and limit to max_solutions_per_problem
    high_quality_solutions.sort(key=lambda x: x["reward"], reverse=True)
    high_quality_solutions = high_quality_solutions[:config.curation.max_solutions_per_problem]
    
    logger.info(f"Found {len(high_quality_solutions)} high-quality solutions for {problem_data['instance_id']} "
                f"(threshold: {config.curation.reward_threshold}, max: {config.curation.max_solutions_per_problem})")
    
    return high_quality_solutions


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
    
    # Process each problem
    all_solutions = []
    problem_stats = defaultdict(int)
    
    for problem_data in tqdm(dataset, desc="Curating problems"):
        try:
            solutions = curate_problem(dict(problem_data), cfg)
            all_solutions.extend(solutions)
            
            # Track statistics
            problem_stats['total_problems'] += 1
            if solutions:
                problem_stats['problems_with_solutions'] += 1
                problem_stats['total_solutions'] += len(solutions)
                problem_stats['max_solutions_per_problem'] = max(
                    problem_stats['max_solutions_per_problem'], len(solutions)
                )
        
        except Exception as e:
            logger.error(f"Failed to process problem {problem_data.get('instance_id', 'unknown')}: {e}")
            continue
    
    # Log statistics
    logger.info(f"Curation completed:")
    logger.info(f"  Total problems processed: {problem_stats['total_problems']}")
    logger.info(f"  Problems with solutions: {problem_stats['problems_with_solutions']}")
    logger.info(f"  Total high-quality solutions: {problem_stats['total_solutions']}")
    logger.info(f"  Average solutions per successful problem: "
                f"{problem_stats['total_solutions'] / max(problem_stats['problems_with_solutions'], 1):.2f}")
    
    if not all_solutions:
        logger.warning("No high-quality solutions found! Check your reward threshold and model setup.")
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
- `solution_diff`: Generated solution diff
- `reward`: Solution quality score (>= {cfg.curation.reward_threshold})
- `messages`: Agent conversation with problem and solution
- `tools`: Shell and navigation tools used by the agent

## Generation Process
- {cfg.curation.num_rollouts_per_problem} rollouts per problem using Nano agent
- Solutions filtered by reward threshold of {cfg.curation.reward_threshold}
- Top {cfg.curation.max_solutions_per_problem} solutions kept per problem
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