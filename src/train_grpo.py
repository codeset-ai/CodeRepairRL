from utils.git import get_commit_hash
from utils.logging import build_html_table
from utils.rewards import xmlcount_reward_func, strict_format_reward_func, correctness_reward_func as correctness_reward_func_original, extract_xml_answer
from utils.gpu_utils import resolve_bf16, resolve_fp16

import os
from typing import Optional
from dataclasses import dataclass, field, MISSING

import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from datasets import load_dataset, Dataset
from peft import LoraConfig as PEFTLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig as HFGRPOConfig, GRPOTrainer


@dataclass
class RunConfig:
    dataset_name: str = "Bobbi/Primevul"
    split: str = "train_paired"
    wandb_project: str = "TTC"
    commit_hash: str = field(default_factory=get_commit_hash)
    train_mode: str = "lora"
    resume_training: bool = False

    def __post_init__(self):
        if self.train_mode not in ["full", "lora"]:
            raise ValueError("train_mode must be either 'full' or 'lora'") 
        
@dataclass
class LoraConfig:  # only used if train_mode == "lora"
    r: int = 32
    lora_alpha: int = 64
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    load_in_4bit: bool = True
    fast_inference: bool = True # Enable vLLM fast inference
    max_seq_length: Optional[int] = 512
    max_lora_rank: Optional[int] = 128
    gpu_memory_utilization: float = 0.6 # Reduce if out of memory

@dataclass
class GRPOConfig:
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.7

    # Optimizer settings
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    
    # Model settings - these will be automatically determined based on GPU architecture
    # when using the custom resolvers in the YAML config
    bf16: bool = MISSING
    fp16: bool = MISSING
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
    run_name: Optional[str] = None
    output_dir: str = "outputs"

@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_grpo_config", node=Config, group="")
OmegaConf.register_resolver("resolve_bf16", resolve_bf16)
OmegaConf.register_resolver("resolve_fp16", resolve_fp16)

TOP_10_CWES = ["CWE-20", "CWE-264", "CWE-200", "CWE-125", "CWE-189", "CWE-416", "CWE-399", "CWE-476", "CWE-362"]  # rempved 119 for class balance

# around 200 tokens
SYSTEM_PROMPT = """
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
"""

def get_primevul(cfg: Config, tokenizer) -> tuple[Dataset, int]:
    data = load_dataset(cfg.run.dataset_name, split=cfg.run.split)
    data = data.filter(lambda x: x["cwe"][0] in TOP_10_CWES and x["is_vulnerable"])  # for simplicity, we only consider the first CWE

    def tokenize_prompt(batch):
        messages = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': func}
            ] for func in batch['func']
        ]
        tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        batch['tokenized_length'] = [len(ids) for ids in tokenized]
        return batch

    # get the tokenized length of the prompt to filter out too long prompts
    data = data.map(tokenize_prompt, batched=True, batch_size=1000)

    print(f"Filtering out prompts longer than {cfg.grpo.max_prompt_length} tokens")
    print(f"Number of prompts before filtering: {len(data)}")
    data = data.filter(lambda x: x['tokenized_length'] <= cfg.grpo.max_prompt_length)
    print(f"Number of prompts after filtering: {len(data)}")

    # now we have guaranteed that SYSTEM_PROMPT + "func" <= max_prompt_length for all examples
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['func']}
        ],
        'answer': x['cwe'][0]
    })
    data = data.shuffle(seed=42)
    return data, max(data['tokenized_length'])

# GRPOTrainer offers no other way to create callbacks with the completions
# quick and dirty solution
def correctness_reward_func(prompts, completions, answer, **kwargs):
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

    return correctness_reward_func_original(prompts, completions, answer, **kwargs)

@hydra.main(version_base="1.1", config_path="conf", config_name="base_grpo_config")
def main(cfg: Config) -> None:
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    # Log precision settings
    precision_mode = "BF16" if cfg.grpo.bf16 else "FP16" if cfg.grpo.fp16 else "FP32"
    print(f"Training with {precision_mode} precision based on GPU architecture")
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if cfg.run.train_mode == "lora":
        lora_config = PEFTLoraConfig(**cfg.lora, task_type = "CAUSAL_LM")
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    dataset, max_prompt_length = get_primevul(cfg, tokenizer)

    if max_prompt_length < cfg.grpo.max_prompt_length:
        diff = cfg.grpo.max_prompt_length - max_prompt_length
        cfg.grpo.max_prompt_length = max_prompt_length
        cfg.grpo.max_completion_length = cfg.grpo.max_completion_length + diff
        assert cfg.grpo.max_completion_length + cfg.grpo.max_prompt_length == tokenizer.model_max_length, "Should fully utilize the model's context window"

    training_args = HFGRPOConfig(**cfg.grpo)

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

    trainer.train(resume_from_checkpoint=cfg.run.resume_training)

    trainer.save_model("grpo_saved_model")

if __name__ == "__main__":
    main() 