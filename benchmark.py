import json
import os
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.primevul import PrimeVul
from src.structured_model_evaluator import StructuredModelEvaluator, BooleanSchema

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("generation_logger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(LOG_DIR, f"generations-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"))
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


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


if __name__ == "__main__":
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    BATCH_SIZE = 8
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MAX_THINKING_TOKENS = 2048
    DO_SAMPLE = True

    logger.info(json.dumps({
        "model": MODEL_NAME,
        "max_thinking_tokens": MAX_THINKING_TOKENS,
        "system_prompt": SYSTEM_PROMPT,
        "adherence_prompt": ADHERENCE_PROMPT,
        "batch_size": BATCH_SIZE,
        "dtype": str(DTYPE),
        "device": DEVICE,
        "do_sample": DO_SAMPLE
    }, indent=4))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE).eval()
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

    with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        prime_tqdm = tqdm(range(0, len(dataset), BATCH_SIZE), desc="PrimeVul")
        for i in prime_tqdm:
            batch = dataset[i:i+BATCH_SIZE]
            prompts, targets = process_batch(batch)
            thoughts, answers = evaluator.generate(prompts, BooleanSchema, max_first_turn_tokens=MAX_THINKING_TOKENS)
            
            num_correct += sum(str(a) == b.answer for a, b in zip(targets, answers))
            prime_tqdm.set_postfix(accuracy=num_correct / (i+len(batch)))

            torch.cuda.empty_cache()  # paranoia, seems to help
            
            for prompt, thought, model_answer, target in zip(prompts, thoughts, answers, targets):
                logger.info(json.dumps({
                    "prompt": prompt,
                    "thought": thought,
                    "model_answer": model_answer.answer,
                    "target": target,
                }, indent=4))
