import logging
from typing import List, Tuple, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer

from src.utils.diff import SearchReplaceDiff, UnifiedDiff


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
Work through the problem here..
</think>
<answer>
The *SEARCH/REPLACE* edits to fix the code in a code block.
</answer>
""".strip()


# Unified diff system prompt alternative
UNIFIED_DIFF_SYSTEM_PROMPT = """
You are a code repair expert tasked with fixing issues in code. You will be provided with:
1. Information about the specific issue (if available)
2. The code segment that needs to be fixed

Your task is to analyze the issue and generate unified diff edits that fix the problem while preserving the code's intended functionality.

Every unified diff edit must use this format:
1. The hunk header showing line numbers: @@ -<original_start>,<original_count> +<new_start>,<new_count> @@
2. Lines starting with '-' to indicate removal
3. Lines starting with '+' to indicate addition 
4. Lines starting with ' ' (space) to indicate context (unchanged lines)

Here is an example:

```
@@ -10,3 +10,3 @@
-    printf("hello\n");
+    printf("Hello world!\n");
     return 0;
```

Please note:
1. Make minimal necessary changes to fix the issue
2. Ensure the fix doesn't break the code's intended functionality
3. Provide proper context lines around your changes
4. If multiple issues exist, provide multiple hunks in your diff

Your response format must follow the template below:
<think>
Work through the problem here..
</think>
<answer>
The unified diff to fix the code in a code block.
</answer>
""".strip()


def repair_single_file_prompt(code: str, description: Optional[str] = None, diff_type: str = "search_replace") -> str:
    """
    Generate a user prompt for code repair of a single file.
    
    Args:
        code: The code to be repaired
        description: Optional description of the issue
        diff_type: Type of diff to use (search_replace or unified)
        
    Returns:
        A formatted user prompt for code repair
    
    TODO: Support multi-file contexts in the future
    """
    prompt = ""
    
    # Add description if provided
    if description:
        prompt += f"--- BEGIN ISSUE DESCRIPTION ---\n{description}\n--- END ISSUE DESCRIPTION ---\n\n"
    
    # Add code (with line numbers for unified diff to help the model)
    if diff_type == "unified":
        numbered_code = "\n".join([f"{i+1:4d} | {line}" for i, line in enumerate(code.splitlines())])
        prompt += f"--- BEGIN CODE ---\n```\n{numbered_code}\n```\n--- END CODE ---\n\n"
        prompt += "Please analyze the code and provide unified diff edits that fix any issues while preserving the code's intended functionality."

    else:
        prompt += f"--- BEGIN CODE ---\n```\n{code}\n```\n--- END CODE ---\n\n"
        prompt += "Please analyze the code and provide search/replace edits that fix any issues while preserving the code's intended functionality."

    
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
    system_prompt: Optional[str] = None,
    diff_type: str = "search_replace"
) -> Tuple[Dataset, int]:
    """
    Create a dataset for code repair tasks from paired before/after code samples.
    
    Args:
        before_codes: List of original code snippets
        after_codes: List of fixed/modified code snippets
        tokenizer: Tokenizer for tokenizing prompts
        descriptions: Optional list of issue descriptions
        max_prompt_length: Maximum prompt length for filtering (default: 512)
        system_prompt: Optional system prompt to use (defaults based on diff_type)
        diff_type: Type of diff to use (search_replace or unified)
        
    Returns:
        Tuple of (processed dataset, maximum token length)
    """
    assert len(before_codes) == len(after_codes), "before_codes and after_codes must have the same length"
    
    if descriptions is not None:
        assert len(before_codes) == len(descriptions), "descriptions must have the same length as code lists"
    else:
        descriptions = [None] * len(before_codes)
    
    # Use appropriate system prompt based on diff type if not provided
    if system_prompt is None:
        system_prompt = CODE_REPAIR_SYSTEM_PROMPT if diff_type == "search_replace" else UNIFIED_DIFF_SYSTEM_PROMPT
    
    # Select the appropriate diff class based on the diff type
    diff_cls = SearchReplaceDiff if diff_type == "search_replace" else UnifiedDiff
    
    # Create dataset items
    data_items = []
    for before, after, desc in zip(before_codes, after_codes, descriptions):
        # Generate diff using the appropriate diff class
        diffs = [diff_cls.from_codes(before, after).to_string()]  # TODO: support multiple diffs
        
        # Generate user prompt
        user_prompt = repair_single_file_prompt(before, desc, diff_type)
        
        # Create dataset item
        item = {
            "before_code": before,
            "after_code": after,
            "description": desc,
            "diffs": diffs,
            "user_prompt": user_prompt,
            "diff_type": diff_type  # Store the diff type for reference during training
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
        "answer": x["diffs"]
    })
    
    # Shuffle dataset
    repair_data = repair_data.shuffle(seed=42)
    
    return repair_data, max(repair_data["tokenized_length"]) 