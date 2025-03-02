import logging
import difflib
from typing import List, Tuple, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


# System prompt for code repair
CODE_REPAIR_SYSTEM_PROMPT = """
You are a code repair expert tasked with fixing issues in code. You will be provided with:
1. Information about the specific issue (if available)
2. The code segment that needs to be fixed

Your task is to analyze the issue and generate *SEARCH/REPLACE* edits that fix the problem while preserving the code's intended functionality.

Every *SEARCH/REPLACE* edit must use this format:
1. The start of search block: <<<<<<< SEARCH
2. A contiguous chunk of lines to search for in the existing source code
3. The dividing line: =======
4. The lines to replace with the fixed implementation
5. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```
<<<<<<< SEARCH
    printf("hello\n");
=======
    printf("Hello world!\n");
>>>>>>> REPLACE
```

Please note:
1. The *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION
2. Make minimal necessary changes to fix the issue
3. Ensure the fix doesn't break the code's intended functionality
4. If multiple issues exist, provide multiple *SEARCH/REPLACE* blocks

Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above.

Your response format must follow the template below:
<think>
Work through the problem here...
</think>
<answer>
The *SEARCH/REPLACE* edits to fix the code in a code block.
</answer>
""".strip()


def generate_search_replace_diff(before_code: str, after_code: str) -> str:
    """
    Generate a SEARCH/REPLACE diff between before and after code versions.
    
    Args:
        before_code: The original code snippet
        after_code: The fixed/modified code snippet
        
    Returns:
        A SEARCH/REPLACE diff representing the changes, focusing on changed chunks
    """
    # Split code into lines
    before_lines = before_code.splitlines()
    after_lines = after_code.splitlines()
    
    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    search_replace_blocks = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # We only care about changes for SEARCH/REPLACE format
        if tag == 'replace':
            search_chunk = '\n'.join(before_lines[i1:i2])
            replace_chunk = '\n'.join(after_lines[j1:j2])
            
            # Create a SEARCH/REPLACE block
            block = "<<<<<<< SEARCH\n"
            block += f"{search_chunk}\n"
            block += "=======\n"
            block += f"{replace_chunk}\n"
            block += ">>>>>>> REPLACE"
            
            search_replace_blocks.append(block)
        
        # For deletions, we just need a SEARCH block with empty REPLACE
        elif tag == 'delete':
            search_chunk = '\n'.join(before_lines[i1:i2])
            
            block = "<<<<<<< SEARCH\n"
            block += f"{search_chunk}\n"
            block += "=======\n"
            block += ">>>>>>> REPLACE"
            
            search_replace_blocks.append(block)
        
        # For insertions, we need context (the line before insertion)
        elif tag == 'insert':
            replace_chunk = '\n'.join(after_lines[j1:j2])
            
            # Need to find the context (the line before insertion)
            context_line = ""
            if i1 > 0:
                context_line = before_lines[i1-1]
            
            block = "<<<<<<< SEARCH\n"
            if context_line:
                block += f"{context_line}\n"
            block += "=======\n"
            if context_line:
                block += f"{context_line}\n"
            block += f"{replace_chunk}\n"
            block += ">>>>>>> REPLACE"
            
            search_replace_blocks.append(block)
    
    # Join all blocks
    return "\n\n".join(search_replace_blocks)


def generate_repair_prompt(code: str, description: Optional[str] = None) -> str:
    """
    Generate a user prompt for code repair.
    
    Args:
        code: The code to be repaired
        description: Optional description of the issue
        
    Returns:
        A formatted user prompt for code repair
    """
    prompt = ""
    
    # Add description if provided
    if description:
        prompt += f"--- BEGIN ISSUE DESCRIPTION ---\n{description}\n--- END ISSUE DESCRIPTION ---\n\n"
    
    # Add code
    prompt += f"--- BEGIN CODE ---\n```\n{code}\n```\n--- END CODE ---\n\n"
    
    # Add instruction
    prompt += "Please analyze the code and provide *SEARCH/REPLACE* edits that fix any issues while preserving the code's intended functionality."
    
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


def create_repair_dataset(
    before_codes: List[str],
    after_codes: List[str],
    tokenizer: PreTrainedTokenizer,
    descriptions: Optional[List[str]] = None,
    max_prompt_length: int = 512,
    system_prompt: Optional[str] = None
) -> Tuple[Dataset, int]:
    """
    Create a dataset for code repair tasks from paired before/after code samples.
    
    Args:
        before_codes: List of original code snippets
        after_codes: List of fixed/modified code snippets
        tokenizer: Tokenizer for tokenizing prompts
        descriptions: Optional list of issue descriptions
        max_prompt_length: Maximum prompt length for filtering (default: 512)
        system_prompt: Optional system prompt to use (defaults to CODE_REPAIR_SYSTEM_PROMPT)
        
    Returns:
        Tuple of (processed dataset, maximum token length)
    """
    assert len(before_codes) == len(after_codes), "before_codes and after_codes must have the same length"
    
    if descriptions is not None:
        assert len(before_codes) == len(descriptions), "descriptions must have the same length as code lists"
    else:
        descriptions = [None] * len(before_codes)
    
    # Use default system prompt if not provided
    if system_prompt is None:
        system_prompt = CODE_REPAIR_SYSTEM_PROMPT
    
    # Create dataset items
    data_items = []
    for i, (before, after, desc) in enumerate(zip(before_codes, after_codes, descriptions)):
        # Generate diff
        diff = generate_search_replace_diff(before, after)
        
        # Generate user prompt
        user_prompt = generate_repair_prompt(before, desc)
        
        # Create dataset item
        item = {
            "before_code": before,
            "after_code": after,
            "description": desc,
            "diff": diff,
            "user_prompt": user_prompt
        }
        data_items.append(item)
    
    # Convert to HF Dataset
    repair_data = Dataset.from_list(data_items)
    
    # Filter by length
    repair_data = filter_by_length(repair_data, tokenizer, system_prompt, max_prompt_length)
    
    # Add prompt field for training
    repair_data = repair_data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["user_prompt"]}
        ],
        "answer": x["diff"]
    })
    
    # Shuffle dataset
    repair_data = repair_data.shuffle(seed=42)
    
    return repair_data, max(repair_data["tokenized_length"]) 