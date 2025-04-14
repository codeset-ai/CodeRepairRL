import logging
from typing import Literal

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

# We should of course use swe-gym instead
def get_swe_bench_repo_repair_dataset(
    split: Literal["train", "dev"] = "dev",
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",  # or any of the other SWE-bench datasets
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset and convert it to a repository repair dataset.
    
    Args:
        split: Dataset split to use ("train" or "dev")
        dataset_name: HuggingFace dataset name for SWE-bench
        
    Returns:
        The processed dataset
    """
    logger.info(f"Loading SWE-bench dataset: {dataset_name}, split: {split}")
    
    # Load the SWE-bench dataset
    swe_ds = load_dataset(dataset_name)[split]
    
    logger.info(f"Creating repository repair dataset with {len(swe_ds)} examples")
    
    return swe_ds




