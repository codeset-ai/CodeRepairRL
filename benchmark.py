import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.primevul import PrimeVul
from src.structured_model_evaluator import StructuredModelEvaluator, BooleanSchema

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about software vulnerabilities."
    "You are given a code snippet and you need to answer whether it is vulnerable or not"
)

def process_batch(items:list[dict])->tuple[list[str], list[bool]]:
    funcs = [x["func"] for x in items]
    funcs = ["Here is a code snippet. Answer whether it is vulnerable or not:\n" + f for f in funcs]
    is_vulnerable = [x["target"] == 0 for x in items]

    return funcs, is_vulnerable


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32

    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()

    evaluator = StructuredModelEvaluator(model, tokenizer, model_name, system_prompt=SYSTEM_PROMPT)

    dataset = PrimeVul(split="test")

    results = []
    num_correct = 0

    with torch.no_grad():
        prime_tqdm = tqdm(range(0, len(dataset), BATCH_SIZE), desc="PrimeVul")
        for i in prime_tqdm:
            batch = dataset[i:i+BATCH_SIZE]
            funcs, is_vulnerable = process_batch(batch)
            answers = evaluator.batch_generate(funcs, BooleanSchema, max_first_turn_tokens=128)

            results.extend(list(zip(funcs, is_vulnerable, [x.answer for x in answers])))
            num_correct += sum(a == bool(b.answer) for a, b in zip(is_vulnerable, answers))
            prime_tqdm.set_postfix(accuracy=num_correct / (i+len(batch)))


    json.dump(results, open("results.json", "w"), indent=4)