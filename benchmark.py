import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from src.primevul import PrimeVul
from src.structured_model_evaluator import StructuredModelEvaluator, BooleanSchema


SYSTEM_PROMPT = (
    "You are a neutral code auditor. For each code snippet provided, objectively assess if it contains vulnerabilities. "
    "If there is any reasonable doubt, weigh both sides before choosing."
)

TASK_PROMPT = (
    "Here is a code snippet. Assess whether it is vulnerable or not."
)

ADHERENCE_PROMPT = (
    "Based on your analysis above, provide your answer in the following JSON format.\n"
    "{\n"
    '    "answer": "True" or "False"\n'
    "}\n"
    "Only output the JSON object, with no additional text before or after."
)  # we also enforce this format programmatically but maybe better to condition the model on it first


def process_batch(items:list[dict])->tuple[list[str], list[bool]]:
    prompts = [TASK_PROMPT + "\n" + x["func"] for x in items]
    targets = [x["target"] == 0 for x in items]

    return prompts, targets


def log_metrics(all_targets, all_predictions):
    """Calculate and log various metrics to wandb"""
    # Convert string values to integers (True -> 1, False -> 0)
    targets_int = [1 if t == "True" else 0 for t in all_targets]
    preds_int = [1 if p == "True" else 0 for p in all_predictions]
    
    precision = precision_score(targets_int, preds_int)
    recall = recall_score(targets_int, preds_int)
    f1 = f1_score(targets_int, preds_int)
    
    wandb.log({
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    # Log the distribution plot to wandb.
    plot_distribution(all_targets, all_predictions)


def plot_distribution(targets, predictions):
    """
    Create a grouped bar chart showing counts for True/False for both target and prediction distributions,
    and log the resulting plot to WandB.
    """
    # Convert "True"/"False" to integers.
    targets_int = [1 if t == "True" else 0 for t in targets]
    preds_int = [1 if p == "True" else 0 for p in predictions]

    # Compute counts for True and False.
    target_true = sum(targets_int)
    target_false = len(targets_int) - target_true
    pred_true = sum(preds_int)
    pred_false = len(preds_int) - pred_true

    # Define categories and counts.
    categories = ['True', 'False']
    target_counts = [target_true, target_false]
    pred_counts = [pred_true, pred_false]

    x = np.arange(len(categories))
    width = 0.35  # width of each bar

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_target = ax.bar(x - width/2, target_counts, width, label='Target', color='dodgerblue')
    bars_pred = ax.bar(x + width/2, pred_counts, width, label='Prediction', color='orange')

    ax.set_ylabel('Count')
    ax.set_title('Distribution of True/False Labels')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Annotate each bar with its count.
    for bar in bars_target + bars_pred:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # vertical offset in points
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    
    # Log the figure as a WandB image.
    wandb.log({"distribution_plot": wandb.Image(fig)})
    plt.close(fig)



if __name__ == "__main__":
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    BATCH_SIZE = 4
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MAX_THINKING_TOKENS = 2048
    DO_SAMPLE = True

    config = {
        "model": MODEL_NAME,
        "max_thinking_tokens": MAX_THINKING_TOKENS,
        "system_prompt": SYSTEM_PROMPT,
        "adherence_prompt": ADHERENCE_PROMPT,
        "batch_size": BATCH_SIZE,
        "dtype": str(DTYPE),
        "device": DEVICE,
        "do_sample": DO_SAMPLE,
        "gpu_name": torch.cuda.get_device_name(DEVICE)
    }

    # Initialize wandb run
    wandb.init(
        project="vulnerability-detection",
        config=config,
        name=f"{MODEL_NAME.split('/')[-1]}-{MAX_THINKING_TOKENS}T-{'sample' if DO_SAMPLE else 'greedy'}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        cache_dir="models"
    ).to(DEVICE).eval()
    torch.compile(model, mode="max-autotune")

    evaluator = StructuredModelEvaluator(
        model,
        tokenizer,
        system_prompt=SYSTEM_PROMPT,
        adherence_prompt=ADHERENCE_PROMPT,
        do_sample=DO_SAMPLE
    )

    dataset = PrimeVul(split="valid")
    num_correct = 0
    all_targets = []
    all_predictions = []
    
    # Create a single table for all predictions
    predictions_table = wandb.Table(
        columns=["code", "thought", "prediction", "target"]
    )
    
    with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        prime_tqdm = tqdm(range(0, len(dataset), BATCH_SIZE), desc="PrimeVul")
        for i in prime_tqdm:
            batch = dataset[i:i+BATCH_SIZE]
            prompts, targets = process_batch(batch)
            thoughts, answers = evaluator.generate(prompts, BooleanSchema, max_first_turn_tokens=MAX_THINKING_TOKENS)
            
            batch_predictions = [b.answer for b in answers]
            all_targets.extend(str(t) for t in targets)
            all_predictions.extend(str(p) for p in batch_predictions)
            
            num_correct += sum(str(a) == b.answer for a, b in zip(targets, answers))
            current_accuracy = num_correct / (i+len(batch))
            prime_tqdm.set_postfix(accuracy=current_accuracy)

            torch.cuda.empty_cache()
            
            # Log metrics for this batch
            wandb.log({"running_accuracy": current_accuracy})
            
            # Add rows to predictions table
            for prompt, thought, model_answer, target in zip(prompts, thoughts, answers, targets):
                code = prompt.split(TASK_PROMPT + "\n")[1]  # Split first, then format
                predictions_table.add_data(
                    code,
                    thought,
                    str(model_answer.answer),
                    str(target)
                )


        wandb.log({"predictions": predictions_table})
        log_metrics(all_targets, all_predictions)
    
    wandb.finish()
