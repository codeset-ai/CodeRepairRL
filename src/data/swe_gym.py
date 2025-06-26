import logging
import hashlib
from typing import Any

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


def _get_swe_gym_split(dataset_name: str, curation_partition: bool, curation_ratio: float = 0.25) -> Dataset:
    """
    Internal function to load and split the SWE-Gym dataset.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        curation_partition: If True, return curation partition; if False, return repo repair partition
        curation_ratio: Ratio of data to allocate to curation partition (default 0.5)
        
    Returns:
        The requested partition of the dataset
    """
    logger.info(f"Loading SWE-bench dataset: {dataset_name}")
    
    # Load the SWE-bench dataset
    swe_ds = load_dataset(dataset_name)["train"]  # only has train split
    
    # Create deterministic split based on instance_id hash
    def should_be_curation(example):
        # Use MD5 hash of instance_id for deterministic splitting
        hash_val = int(hashlib.md5(example['instance_id'].encode()).hexdigest(), 16)
        # Convert to [0, 1] range and compare with curation_ratio
        return (hash_val / (16**32)) < curation_ratio
    
    # Filter based on partition type
    if curation_partition:
        swe_ds = swe_ds.filter(should_be_curation)
        logger.info(f"Creating curation dataset with {len(swe_ds)} examples")
    else:
        swe_ds = swe_ds.filter(lambda x: not should_be_curation(x))
        logger.info(f"Creating repository repair dataset with {len(swe_ds)} examples")
    
    # Add a dummy "prompt" key for compatibility with trl
    swe_ds = swe_ds.map(lambda x: {"prompt": "Dummy"})
    
    return swe_ds

# mirroring the other data methods though not strictly doing much
def get_swe_gym_repo_repair_dataset(
    dataset_name: str = "SWE-Gym/SWE-Gym-Lite",
    curation_ratio: float = 0.25,
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset and convert it to a repository repair dataset.
    This function returns the non-curation partition of the data for RL/GRPO training.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        curation_ratio: Ratio of data to allocate to curation partition (default 0.5)
        
    Returns:
        The processed dataset (repo repair partition)
    """
    return _get_swe_gym_split(dataset_name, curation_partition=False, curation_ratio=curation_ratio)

def get_swe_gym_curation_dataset(
    dataset_name: str = "SWE-Gym/SWE-Gym-Lite",
    curation_ratio: float = 0.25,
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset for SFT data curation via rejection sampling.
    This function returns the curation partition of the data, ensuring no overlap 
    with the repository repair dataset. Used by curate_sft_data.py to generate
    high-quality SFT examples through multiple rollouts and filtering.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        curation_ratio: Ratio of data to allocate to curation partition (default 0.5)
        
    Returns:
        The processed dataset (curation partition for rejection sampling)
    """
    return _get_swe_gym_split(dataset_name, curation_partition=True, curation_ratio=curation_ratio)

def format_conversation_for_sft(example: dict[str, Any], tokenizer) -> dict[str, str]:
    """
    Format a curated example into a conversation for SFT training.
    Expects 'messages' and 'tools' fields to exist.
    """
    # Apply chat template with messages and tools - FAIL if these don't exist
    formatted_text = tokenizer.apply_chat_template(
        example["messages"], 
        tools=example["tools"],
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}

def get_swe_gym_formatted_sft_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_length: int = 8192,
    reward_min: float = 0.2,
    **kwargs
) -> Dataset:
    """
    Load and format a curated SFT dataset for training.
    This function loads an already-curated dataset (created by curate_sft_data.py)
    and formats it for SFT training.
    
    Args:
        dataset_name: HuggingFace dataset name for curated SFT data
        tokenizer: Tokenizer for formatting conversations
        max_seq_length: Maximum sequence length for filtering
        
    Returns:
        The formatted dataset ready for SFT training
    """
    logger.info(f"Loading curated SFT dataset: {dataset_name}")
    
    # Load the curated dataset
    dataset = load_dataset(dataset_name, split="train")
    
    logger.info(f"Preparing dataset with {len(dataset)} examples...")
    
    # Format conversations
    dataset = dataset.map(
        lambda x: format_conversation_for_sft(x, tokenizer),
        desc="Formatting conversations"
    )
    
    # Filter out examples that are too long
    def filter_length(example):
        tokens = tokenizer.encode(example["text"])
        return len(tokens) <= max_seq_length
    
    original_size = len(dataset)
    dataset = dataset.filter(filter_length, desc="Filtering by length")
    dataset = dataset.filter(lambda x: x["reward"] > reward_min)
    logger.info(f"Filtered dataset from {original_size} to {len(dataset)} examples "
                f"(max_seq_length={max_seq_length})")
    
    return dataset

if __name__ == "__main__":
    ds = load_dataset("SWE-Gym/SWE-Gym-Lite")
    print(ds)
    
    # Test the split functions
    curation_ds = get_swe_gym_curation_dataset()
    repair_ds = get_swe_gym_repo_repair_dataset()
    
    print(f"Curation dataset size: {len(curation_ds)}")
    print(f"Repo repair dataset size: {len(repair_ds)}")
    print(f"Total: {len(curation_ds) + len(repair_ds)}")
    
    # Verify no overlap
    curation_ids = set(curation_ds['instance_id'])
    repair_ids = set(repair_ds['instance_id'])
    overlap = curation_ids.intersection(repair_ids)
    print(f"Overlap between partitions: {len(overlap)} items")