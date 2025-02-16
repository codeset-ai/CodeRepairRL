from utils.logging import build_html_table

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # important to call this first

import os
import re
from typing import Optional
from dataclasses import dataclass, field

import wandb
import hydra
from hydra.core.config_store import ConfigStore

from datasets import load_dataset, Dataset
from trl import GRPOConfig as HFGRPOConfig, GRPOTrainer


@dataclass
class RunConfig:
    dataset_name: str = "Bobbi/Primevul"
    wandb_project: str = "TTC"
    torch_dtype: str = "bfloat16"

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    load_in_4bit: bool = True
    fast_inference: bool = True # Enable vLLM fast inference
    max_seq_length: Optional[int] = 512
    max_lora_rank: Optional[int] = 32
    gpu_memory_utilization: float = 0.6 # Reduce if out of memory

@dataclass
class GRPOConfig:
    use_vllm: bool = True

    # Optimizer settings
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    
    # Model settings
    bf16: bool = True
    fp16: bool = False
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # Generation settings
    num_generations: int = 6
    max_prompt_length: int = 256
    max_completion_length: int = 200

    # Training loop settings
    logging_steps: int = 1
    max_steps: int = 250
    save_steps: int = 250
    max_grad_norm: float = 0.1
    
    # Logging settings
    report_to: str = "wandb"
    run_name: str = "Qwen2.5-Coder-1.5B-Instruct-GRPO-Primevul"
    output_dir: str = "outputs"

@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="grpo_config", node=Config, group="")


SYSTEM_PROMPT = """
Your are a neutral code auditor detecting software vulnerabilities. Do not provide fixes or code solutions - focus only on vulnerability detection.
Respond in the following format:
<think>
...
</think>
<answer>
True or False
</answer>
"""

def extract_xml_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()

    return "N/A"

def get_primevul(cfg: RunConfig, split="train") -> Dataset:
    data = load_dataset(cfg.dataset_name, split=split)
    data = data.filter(lambda x: len(x['func']) < 2200)  # slightly less than 1024, guestimated that max_prompt+system_prompt ~= 1024
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['func']}
        ],
        'answer': str(x['is_vulnerable'])
    })
    return data

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # For each sample, add a row to our HTML table.
    html_rows = []
    for prompt_item, response, ext, ans in zip(prompts, responses, extracted_responses, answer):
         prompt_text = prompt_item[-1]['content'] if prompt_item else ""
         html_rows.append((prompt_text, response, ext, ans))
    # Rebuild and log the HTML table.
    html_table = build_html_table(html_rows)
    wandb.log({"eval_table": wandb.Html(html_table)})
    return [2.0 if ext == a else 0.0 for ext, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n([\s\S]+?)\n</think>\n<answer>\n([^\n]+)\n</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return max(count, 0.0)

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def test_inference(model, tokenizer):
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    # Test prompt
    prompt = """
    int main() {
        char buffer[10];
        gets(buffer);
        return 0;
    }
    """
    text = tokenizer.apply_chat_template([
        {"role" : "user", "content" : prompt},
    ], tokenize = False, add_generation_prompt = True)
        
    # Format input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    # Generate without fine-tuning
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text
    print("\nOutput without fine-tuning:")
    print(output)

    # Generate with fine-tuning
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text
    print("\nOutput with fine-tuning:")
    print(output)

@hydra.main(version_base="1.1", config_path="conf", config_name="grpo_config")
def main(cfg: Config) -> None:
    PatchFastRL("GRPO", FastLanguageModel)

    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    model, tokenizer = FastLanguageModel.from_pretrained(**cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.model.max_lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = cfg.model.max_lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    training_args = HFGRPOConfig(**cfg.grpo)

    dataset = get_primevul(cfg.run)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()

    # Save the trained model
    trainer.save_model("grpo_saved_model")

    # Test inference
    test_inference(model, tokenizer)

if __name__ == "__main__":
    main() 