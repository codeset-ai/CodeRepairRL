import logging
from typing import List, Tuple, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


# Generic system prompt for code classification
CODE_CLASSIFICATION_SYSTEM_PROMPT = """You are a code classification expert tasked with identifying issues in code. You will be provided with:
1. Information about the classification task (if available)
2. The code segment that needs to be classified

Your task is to analyze the code and assign the most appropriate label from the provided list.

Respond in the following format:
<think>
Work through the problem here...
</think>
<answer>
one label from the list
</answer>
""".strip()


def generate_classification_prompt(code: str, description: Optional[str] = None) -> str:
    """
    Generate a user prompt for code classification.
    
    Args:
        code: The code to be classified
        description: Optional description of the classification task
        
    Returns:
        A formatted user prompt for code classification
    """
    prompt = ""
    
    # Add description if provided
    if description:
        prompt += f"--- BEGIN TASK DESCRIPTION ---\n{description}\n--- END TASK DESCRIPTION ---\n\n"
    
    # Add code
    prompt += f"--- BEGIN CODE ---\n```\n{code}\n```\n--- END CODE ---\n\n"
    
    # Add instruction
    prompt += "Please analyze the code and provide the most appropriate classification label."
    
    return prompt.strip()


def filter_by_length(data, tokenizer, system_prompt, max_prompt_length: int, user_content_key: str = "user_prompt"):
    """
    Filter dataset by tokenized prompt length.
    
    Args:
        data: Dataset to filter
        tokenizer: Tokenizer for tokenizing prompts
        system_prompt: System prompt to use
        max_prompt_length: Maximum allowed token length for prompts
        user_content_key: Key in the dataset items to use as user content
        
    Returns:
        Filtered dataset
    """
    def tokenize_prompt(batch):
        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item}
            ] for item in batch[user_content_key]
        ]
        tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        batch["tokenized_length"] = [len(ids) for ids in tokenized]
        return batch

    # Get tokenized lengths and filter long prompts
    data = data.map(tokenize_prompt, batched=True, batch_size=1000)

    logger.info(f"Filtering out prompts longer than {max_prompt_length} tokens")
    logger.info(f"Number of prompts before filtering: {len(data)}")
    data = data.filter(lambda x: x["tokenized_length"] <= max_prompt_length)
    logger.info(f"Number of prompts after filtering: {len(data)}")
    
    return data


def create_classification_dataset(
    codes: List[str],
    labels: List[str],
    tokenizer: PreTrainedTokenizer,
    descriptions: Optional[List[str]] = None,
    max_prompt_length: int = 512,
    system_prompt: Optional[str] = None
) -> Tuple[Dataset, int]:
    """
    Create a dataset for code classification tasks.
    
    Args:
        codes: List of code snippets to classify
        labels: List of classification labels for each code snippet
        descriptions: Optional list of task descriptions
        tokenizer: Tokenizer for tokenizing prompts
        max_prompt_length: Maximum prompt length for filtering (default: 512)
        system_prompt: Optional system prompt to use (defaults to CODE_CLASSIFICATION_SYSTEM_PROMPT)
        
    Returns:
        Tuple of (processed dataset, maximum token length)
    """
    assert len(codes) == len(labels), "codes and labels must have the same length"
    
    if descriptions is not None:
        assert len(codes) == len(descriptions), "descriptions must have the same length as code list"
    else:
        descriptions = [None] * len(codes)
    
    # Use default system prompt if not provided
    if system_prompt is None:
        system_prompt = CODE_CLASSIFICATION_SYSTEM_PROMPT
    
    # Create dataset items
    data_items = []
    for i, (code, label, desc) in enumerate(zip(codes, labels, descriptions)):
        # Generate user prompt
        user_prompt = generate_classification_prompt(code, desc)
        
        # Create dataset item
        item = {
            "code": code,
            "label": label,
            "description": desc,
            "user_prompt": user_prompt
        }
        data_items.append(item)
    
    # Convert to HF Dataset
    classification_data = Dataset.from_list(data_items)
    
    # Filter by length
    classification_data = filter_by_length(classification_data, tokenizer, system_prompt, max_prompt_length)
    
    # Add prompt field for training
    classification_data = classification_data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["user_prompt"]}
        ],
        "answer": x["label"]
    })
    
    # Shuffle dataset
    classification_data = classification_data.shuffle(seed=42)
    
    return classification_data, max(classification_data["tokenized_length"]) 