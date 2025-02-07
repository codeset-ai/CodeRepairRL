import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.primevul import PrimeVul
from src.structured_model_evaluator import StructuredModelEvaluator, BooleanSchema


SYSTEM_PROMPT = (
    "You are a neutral code auditor. For each code snippet provided, objectively assess if it contains vulnerabilities. "
    "If there is any reasonable doubt, weigh both sides before choosing."
)

ADHERENCE_PROMPT = (
    "Based on your analysis above, decide if the code snippet is vulnerable."
    "Then, reply exactly with the JSON format where 'answer' is either 'True' or 'False'."
)

def process_batch(items:list[dict])->tuple[list[str], list[bool]]:
    funcs = [x["func"] for x in items]
    funcs = ["Here is a code snippet. Answer whether it is vulnerable or not:\n" + f for f in funcs]
    is_vulnerable = [x["target"] == 0 for x in items]

    return funcs, is_vulnerable


if __name__ == "__main__":
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    BATCH_SIZE = 4

    # model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).to(DEVICE).eval()
    torch.compile(model, mode="max-autotune")

    evaluator = StructuredModelEvaluator(
        model,
        tokenizer,
        system_prompt=SYSTEM_PROMPT,
        adherence_prompt=ADHERENCE_PROMPT,
        do_sample=True
    )

    dataset = PrimeVul(split="test")

    results = []
    num_correct = 0

    with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        prime_tqdm = tqdm(range(0, len(dataset), BATCH_SIZE), desc="PrimeVul")
        for i in prime_tqdm:
            batch = dataset[i:i+BATCH_SIZE]
            funcs, is_vulnerable = process_batch(batch)
            answers = evaluator.generate(funcs, BooleanSchema, max_first_turn_tokens=1024)

            results.extend(list(zip(funcs, is_vulnerable, [x.answer for x in answers])))
            num_correct += sum(str(a) == b.answer for a, b in zip(is_vulnerable, answers))
            prime_tqdm.set_postfix(accuracy=num_correct / (i+len(batch)))

            torch.cuda.empty_cache()  # paranoia


    json.dump(results, open("results.json", "w"), indent=4)