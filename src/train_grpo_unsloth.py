import os
import logging
from functools import partial

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # important to call this first

import hydra
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
from src.train_grpo import Config
from src.data import get_stack_repair_dataset, get_primevul_repair_dataset, get_primevul_detection_dataset

logger = logging.getLogger(__name__)



@hydra.main(version_base="1.1", config_path="conf", config_name="base_grpo_config")
def main(cfg: Config) -> None:
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    model, tokenizer = FastLanguageModel.from_pretrained(**cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if cfg.run.train_mode == "lora":
        model = FastLanguageModel.get_peft_model(
            model,
            **cfg.lora,
            use_gradient_checkpointing = "unsloth", # Enable long context finetuning
            random_state = 3407,
        )
    model.print_trainable_parameters()

    # Get dataset based on the task
    if cfg.run.task == "repair":
        repair_dataset = get_stack_repair_dataset if cfg.run.dataset_type == "stack" else get_primevul_repair_dataset
        dataset, max_prompt_length = repair_dataset(
            tokenizer=tokenizer,
            max_prompt_length=cfg.grpo.max_prompt_length,
            diff_type=cfg.run.diff_type  # Pass the diff type from config
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            partial(diff_format_reward_func, diff_type=cfg.run.diff_type),  # we need to know the type of diff to use to process the output    
            partial(diff_similarity_reward_func, diff_type=cfg.run.diff_type),
        ]
    elif cfg.run.task == "detection":  # primevul only
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
    else:
        raise ValueError(f"Unknown task: {cfg.run.task}")

    # Adjust sequence lengths if needed
    if max_prompt_length < cfg.grpo.max_prompt_length:
        diff = cfg.grpo.max_prompt_length - max_prompt_length
        cfg.grpo.max_prompt_length = max_prompt_length
        cfg.grpo.max_completion_length = cfg.grpo.max_completion_length + diff


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
    model_save_path = f"grpo_{cfg.run.task}_unsloth_model"
    trainer.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main() 