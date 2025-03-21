import logging
from transformers import AutoTokenizer
from src.data.stack import get_stack_repair_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Verify the stack repair dataset functionality."""
    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-mono")
    
    # Get a small sample dataset
    dataset, max_tokens = get_stack_repair_dataset(
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_samples=10,
        min_docstring_length=50,
        diff_type="search_replace"  # or "unified"
    )
    
    # Log dataset info
    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Maximum token length: {max_tokens}")
    
    # Display a few examples
    if len(dataset) > 0:
        logger.info("Example of the first dataset entry:")
        example = dataset[0]
        
        print("\nUser prompt:")
        print("=" * 80)
        print(example["user_prompt"])
        print("=" * 80)
        
        print("\nExpected answer (diff):")
        print("=" * 80)
        print(example["diff"])
        print("=" * 80)
        
        # Print the prompt field which is formatted for model training
        print("\nFormatted prompt for model training:")
        logger.info(f"System prompt length: {len(example['prompt'][0]['content'])}")
        logger.info(f"User prompt length: {len(example['prompt'][1]['content'])}")
        
        # Show token counts
        logger.info(f"Tokenized length: {example['tokenized_length']}")
    else:
        logger.warning("No samples were found in the dataset")

if __name__ == "__main__":
    main() 