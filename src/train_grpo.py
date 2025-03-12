import os
import logging
from typing import Optional
from functools import partial
from dataclasses import dataclass, field

import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from peft import LoraConfig as PEFTLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig as HFGRPOConfig, GRPOTrainer as HFGRPOTrainer

from src.utils.rewards import (
    # reasoning rewards
    partial_reasoning_format_reward_func,
    strict_reasoning_format_reward_func,
    # detection rewards
    correctness_reward_func,
    # repair rewards
    diff_format_reward_func,
    diff_similarity_reward_func,
)
from src.utils.resolvers import resolve_bf16, resolve_fp16, resolve_git_commit_hash
from src.data.repairllama import get_repairllama_dataset
from src.data.primevul import get_primevul_repair_dataset, get_primevul_detection_dataset

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    wandb_project: str = "TTC"
    train_mode: str = "lora"  # "full" or "lora"
    task: str = "repair"  # "detection" or "repair"
    dataset_type: str = "primevul"  # "primevul" or "repairllama"
    diff_type: str = "search_replace"  # "search_replace" or "unified" (for repair)
    commit_hash: str = ""  # added at runtime
    resume_training: bool = False

    def __post_init__(self):
        if self.train_mode not in ["full", "lora"]:
            raise ValueError("train_mode must be either 'full' or 'lora'")
        if self.task not in ["detection", "repair"]:
            raise ValueError("task must be either 'detection' or 'repair'")
        if self.dataset_type not in ["primevul", "repairllama"]:
            raise ValueError("dataset_type must be either 'repairllama', or 'primevul'")
        if self.diff_type not in ["search_replace", "unified"]:
            raise ValueError("diff_type must be either 'search_replace', or 'unified'")
        
@dataclass
class LoraConfig:  # only used if train_mode == "lora"
    r: int = 32
    lora_alpha: int = 64
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

@dataclass
class ModelConfig:  # mostly unsloth-specific settings
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
    lr_scheduler_type: str = "linear"  # Cosine learning rates have been shown to be bad for GRPO, see discussion: https://x.com/kalomaze/status/1895549497692090573
    optim: str = "paged_adamw_8bit"
    
    # Model settings - these will be automatically determined based on GPU architecture
    # when using the custom resolvers in the YAML config
    bf16: bool = False
    fp16: bool = False 
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    # Generation settings
    num_generations: int = 4
    max_prompt_length: int = 256
    max_completion_length: int = 256

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
OmegaConf.register_resolver("resolve_git_commit_hash", resolve_git_commit_hash)


@hydra.main(version_base="1.1", config_path="conf", config_name="base_grpo_config")
def main(cfg: Config) -> None:
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    # Log precision settings
    precision_mode = "BF16" if cfg.grpo.bf16 else "FP16" if cfg.grpo.fp16 else "FP32"
    logger.info(f"Training with {precision_mode} precision based on GPU architecture")
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # by padding a batch of prompts on the left side we can generate many completions in parallel (padding tokens are masked away)

    if cfg.run.train_mode == "lora":
        lora_config = PEFTLoraConfig(**cfg.lora, task_type = "CAUSAL_LM")
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Get dataset based on the task
    if cfg.run.task == "repair":
        repair_dataset = get_primevul_repair_dataset if cfg.run.dataset_type == "primevul" else get_repairllama_dataset
        dataset, max_prompt_length = repair_dataset(
            tokenizer,
            cfg.grpo.max_prompt_length
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            partial(diff_format_reward_func, diff_type=cfg.grpo.diff_type),  # we need to know the type of diff to use to process the output    
            partial(diff_similarity_reward_func, diff_type=cfg.grpo.diff_type),
        ]
    elif cfg.run.task == "detection":  # primevul only
        dataset, max_prompt_length = get_primevul_detection_dataset(
            tokenizer, 
            cfg.grpo.max_prompt_length
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            correctness_reward_func,
        ]
    else:
        raise ValueError(f"Unknown task: {cfg.run.task}")  # can't happen but looks nice

    # Adjust sequence lengths if needed (ensures we are not wasting context window)
    if max_prompt_length < cfg.grpo.max_prompt_length:
        diff = cfg.grpo.max_prompt_length - max_prompt_length
        cfg.grpo.max_prompt_length = max_prompt_length
        cfg.grpo.max_completion_length = cfg.grpo.max_completion_length + diff
        assert cfg.grpo.max_completion_length + cfg.grpo.max_prompt_length == tokenizer.model_max_length, "Should fully utilize the model's context window"

    training_args = HFGRPOConfig(**cfg.grpo)

    # Initialize trainer with task-specific reward functions
    trainer = HFGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=cfg.run.resume_training)

    # Save with task-specific name
    model_save_path = f"grpo_{cfg.run.task}_model"
    trainer.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main() 