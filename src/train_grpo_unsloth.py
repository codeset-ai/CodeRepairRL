from utils.rewards import xmlcount_reward_func, strict_format_reward_func
from train_grpo import Config, get_primevul, correctness_reward_func

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # important to call this first

import os

import hydra
from hydra.core.config_store import ConfigStore

from datasets import load_dataset, Dataset
from trl import GRPOConfig as HFGRPOConfig, GRPOTrainer



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

    dataset, max_prompt_length = get_primevul(cfg, tokenizer)

    if max_prompt_length < cfg.grpo.max_prompt_length:
        diff = cfg.grpo.max_prompt_length - max_prompt_length
        cfg.grpo.max_prompt_length = max_prompt_length
        cfg.grpo.max_completion_length = cfg.grpo.max_completion_length + diff
        assert cfg.grpo.max_completion_length + cfg.grpo.max_prompt_length == cfg.model.max_seq_length

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

    trainer.train()

    trainer.save_model("grpo_saved_model")

if __name__ == "__main__":
    main() 