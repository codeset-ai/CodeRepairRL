import logging

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

# mirroring the other data methods though not strictly doing much
def get_swe_gym_repo_repair_dataset(
    dataset_name: str = "SWE-Gym/SWE-Gym-Lite",
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset and convert it to a repository repair dataset.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        
    Returns:
        The processed dataset
    """
    logger.info(f"Loading SWE-bench dataset: {dataset_name}")
    
    # Load the SWE-bench dataset
    swe_ds = load_dataset(dataset_name)["train"]  # only has train split
    
    logger.info(f"Creating repository repair dataset with {len(swe_ds)} examples")
    
    return swe_ds

if __name__ == "__main__":
    ds = load_dataset("SWE-Gym/SWE-Gym-Lite")
    print(ds)