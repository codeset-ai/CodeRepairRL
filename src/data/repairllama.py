import logging
from typing import Tuple, Optional

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

from src.data.code_repair import create_repair_dataset


logger = logging.getLogger(__name__)


def get_repairllama_dataset(    
    tokenizer: PreTrainedTokenizer,
    split: str = "ir1xor1",
    max_prompt_length: int = 512,
    system_prompt: Optional[str] = None
) -> Tuple[Dataset, int]:
    """
    Get the RepairLLaMA dataset for code repair.
    
    Args:
        tokenizer: Tokenizer for tokenizing prompts
        max_prompt_length: Maximum prompt length for filtering (default: 512)
        split: Dataset split to use
        system_prompt: Optional system prompt to override the one in code_repair.py
        
    Returns:
        Tuple of (processed dataset, maximum token length)
    """
    # Load the dataset
    dataset = load_dataset("ASSERT-KTH/repairllama-datasets", split=split)
    
    # Extract the buggy and fixed code
    buggy_codes = []
    fixed_codes = []
    descriptions = []
    
    for item in dataset:
        if "input" in item and "output" in item:
            buggy_codes.append(item["input"])
            fixed_codes.append(item["output"])
            
            # Use a generic description since metadata is not available
            descriptions.append("Fix the bug in this code.")
    
    # Create the repair dataset
    return create_repair_dataset(
        before_codes=buggy_codes,
        after_codes=fixed_codes,
        descriptions=descriptions,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        system_prompt=system_prompt
    ) 