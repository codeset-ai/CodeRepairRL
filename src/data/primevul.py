import logging
from typing import Tuple, Optional

from datasets import load_dataset, Dataset

from src.data.code_repair import create_repair_dataset


logger = logging.getLogger(__name__)


def get_primevul_repair_dataset(
    dataset_name: str = "ASSERT-KTH/PrimeVul",
    split: str = "train_paired",
    tokenizer = None,
    max_prompt_length: Optional[int] = None,
    system_prompt: Optional[str] = None
) -> Tuple[Dataset, int]:
    """
    Create a dataset for code repair tasks from PrimeVul paired data.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        split: Dataset split to use (must be paired)
        tokenizer: Tokenizer for tokenizing prompts
        max_prompt_length: Maximum allowed token length for prompts
        system_prompt: Optional system prompt to use
        
    Returns:
        Tuple of (processed dataset, maximum token length)
    """
    assert split.endswith("_paired"), "Only paired dataset is supported for repair tasks"

    # Load the dataset
    data = load_dataset(dataset_name, split=split)
    data = data.filter(lambda x: "Other" not in x["cwe"])  # as we do not have a description for "Other"
    
    # Group by hash to find pairs (vulnerable and fixed versions)
    # In paired dataset, adjacent items have the same hash but different is_vulnerable status
    all_items = list(data)
    
    before_codes = []
    after_codes = []
    descriptions = []

    for i in range(0, len(all_items), 2):
        if i + 1 >= len(all_items):
            break
            
        fixed = all_items[i]
        vulnerable = all_items[i + 1]
        
        if not fixed["hash"] == vulnerable["hash"] or not fixed["is_vulnerable"] == (not vulnerable["is_vulnerable"]):
            logger.warning(f"Skipping pair at index {i} due to mismatch in hash or vulnerability status")
            continue
        
        before_codes.append(vulnerable["func"])
        after_codes.append(fixed["func"])
        descriptions.append(f"Type: {vulnerable['cwe']}\nDescription: {vulnerable['cwe_description']}")
    
    # Create the repair dataset using the generalized function
    return create_repair_dataset(
        before_codes=before_codes,
        after_codes=after_codes,
        descriptions=descriptions,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        system_prompt=system_prompt
    )
