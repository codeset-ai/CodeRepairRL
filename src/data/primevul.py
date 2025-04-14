from typing import Optional

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

from src.data.code_mono_repair import create_mono_repair_dataset
from src.data.code_classification import create_classification_dataset


def get_primevul_repair_dataset(
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int = 512,
    split: str = "train_paired",
    system_prompt: Optional[str] = None,
    context_lines: int = 0
) -> Dataset:
    """
    Create a dataset for code repair tasks from PrimeVul paired data.
    
    Args:
        tokenizer: Tokenizer for tokenizing prompts
        max_prompt_length: Maximum prompt length for filtering (default: 512)
        dataset_name: Name of the dataset on HuggingFace Hub
        split: Dataset split to use (must be paired)
        system_prompt: Optional system prompt to use
        context_lines: Number of context lines to include in diffs (default: 0)
    Returns:
        The processed dataset
    """
    assert split.endswith("_paired"), "Only paired dataset is supported for repair tasks"

    # Load the dataset
    data = load_dataset("ASSERT-KTH/PrimeVul", split=split).shuffle(seed=42)
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
            raise ValueError(f"Pair at index {i} has mismatching hash or vulnerability status")
        
        before_codes.append(vulnerable["func"])
        after_codes.append(fixed["func"])
        descriptions.append(f"Type: {vulnerable['cwe']}\nDescription: {vulnerable['cwe_description']}")
    
    # Create the repair dataset using the generalized function
    return create_mono_repair_dataset(
        before_codes=before_codes,
        after_codes=after_codes,
        descriptions=descriptions,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        system_prompt=system_prompt,
        context_lines=context_lines
    )


# List of top CWEs to focus on
TOP_CWES = ["CWE-20", "CWE-264", "CWE-200", "CWE-125", "CWE-189", "CWE-416", "CWE-399", "CWE-476", "CWE-362"]

# System prompt for vulnerability detection
VULN_DETECTION_SYSTEM_PROMPT = """
You are a code auditor identifying software vulnerabilities, or lack thereof, without offering fixes.
Use only these labels (description provided for context):
    - CWE-20: Poor input validation allows malicious data.
    - CWE-264: Weak access controls enable unauthorized actions.
    - CWE-200: Unintended sensitive data exposure.
    - CWE-125: Out-of-bounds read leaks data.
    - CWE-189: Numeric errors cause calculation and logic faults.
    - CWE-416: Use-after-free—accessing deallocated memory.
    - CWE-399: Mismanaged resources leading to leaks/exhaustion.
    - CWE-476: Null pointer dereference results in crashes.
    - CWE-362: Race conditions from unsynchronized concurrent operations.
Respond in the following format:
<think>
...
</think>
<answer>
one label from the list
</answer>
""".strip()

def get_primevul_detection_dataset(
    tokenizer: PreTrainedTokenizer,
    split: str = "train_paired",
    max_prompt_length: int = 512,
) -> Dataset:
    """
    Create a dataset for vulnerability detection tasks from PrimeVul data.
    
    Args:
        tokenizer: Tokenizer for tokenizing prompts
        split: Dataset split to use
        max_prompt_length: Maximum prompt length for filtering (default: 512)
        
    Returns:
        The processed dataset
    """
    # Load the dataset
    data = load_dataset("ASSERT-KTH/PrimeVul", split=split)
    # Filter to only include vulnerable samples with CWEs in our target list
    data = data.filter(lambda x: x["is_vulnerable"] and x["cwe"][0] in TOP_CWES)
    
    # Extract the code and labels
    codes = [item["func"] for item in data]
    labels = [item["cwe"][0] for item in data]

    # Create the classification dataset
    return create_classification_dataset(
        codes=codes,
        labels=labels,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        system_prompt=VULN_DETECTION_SYSTEM_PROMPT
    )