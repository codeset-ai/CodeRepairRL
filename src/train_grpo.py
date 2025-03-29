import os
import logging
from typing import Optional
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
from src.utils.resolvers import resolve_git_commit_hash
from src.data import get_stack_repair_dataset, get_primevul_repair_dataset, get_primevul_detection_dataset

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    wandb_project: str = "TTC"
    lora: bool = True
    task_type: str = "repair"  # "detection" or "repair"
    dataset_type: str = "stack"  # "primevul" or "stack"
    context_lines: int = 0  # number of context lines to include in diffs
    commit_hash: str = ""  # added at runtime
    resume_training: bool = False

    def __post_init__(self):
        if self.task_type not in ["detection", "repair"]:
            raise ValueError("task_type must be either 'detection' or 'repair'")
        if self.dataset_type not in ["primevul", "stack"]:
            raise ValueError("dataset_type must be either 'stack', or 'primevul'")

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
     # only used if run.lora is true
    r: int = 32
    lora_alpha: int = 64
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

@dataclass
class GRPOConfig:
    # vLLM generation settings
    use_vllm: bool = True
    
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
    bf16: bool = True
    fp16: bool = False 
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # Reward settings
    scale_rewards: bool = False  # from Dr. GRPO, reward scaling introduces question-level difficulty bias
    
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
    log_completions: bool = True

@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_grpo_config", node=Config, group="")
OmegaConf.register_new_resolver("resolve_git_commit_hash", resolve_git_commit_hash)


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    # Log precision settings
    precision_mode = "BF16" if cfg.grpo.bf16 else "FP16" if cfg.grpo.fp16 else "FP32"
    logger.info(f"Training with {precision_mode} precision based on GPU architecture")
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # by padding a batch of prompts on the left side we can generate many completions in parallel (padding tokens are masked away)

    if cfg.run.lora:
        lora_params = OmegaConf.to_container(cfg.model, resolve=True)
        lora_config = PEFTLoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["lora_alpha"],
            target_modules=lora_params["target_modules"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Get dataset based on the task
    if cfg.run.task_type == "repair":
        repair_dataset = get_stack_repair_dataset if cfg.run.dataset_type == "stack" else get_primevul_repair_dataset
        dataset, max_prompt_length = repair_dataset(
            tokenizer=tokenizer,
            max_prompt_length=cfg.grpo.max_prompt_length,
            context_lines=cfg.run.context_lines
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            diff_format_reward_func,
            diff_similarity_reward_func, 
        ]
        reward_weights = [0.1, 0.2, 0.3, 0.4]
    elif cfg.run.task_type == "detection":  # primevul only
        if cfg.run.dataset_type == "stack": raise ValueError("Stack does not support detection task")
        dataset, max_prompt_length = get_primevul_detection_dataset(
            tokenizer=tokenizer, 
            max_prompt_length=cfg.grpo.max_prompt_length
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            correctness_reward_func,
        ]
        reward_weights = [0.1, 0.2, 0.7]
    else:
        raise ValueError(f"Unknown task: {cfg.run.task_type}")  # can't happen but looks nice

    # Adjust sequence lengths if needed (ensures we are not wasting context window)
    if max_prompt_length < cfg.grpo.max_prompt_length:
        diff = cfg.grpo.max_prompt_length - max_prompt_length
        cfg.grpo.max_prompt_length = max_prompt_length
        cfg.grpo.max_completion_length = cfg.grpo.max_completion_length + diff

    # Convert grpo config from OmegaConf to regular Python dict to ensure JSON serialization works
    grpo_params = OmegaConf.to_container(cfg.grpo, resolve=True)
    grpo_params["reward_weights"] = reward_weights
    training_args = HFGRPOConfig(**grpo_params)

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
    model_save_path = f"grpo_{cfg.run.task_type}_model"
    trainer.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main() 