import json
import os
import random
import logging
import contextlib
from typing import Literal
from functools import wraps
from datetime import datetime

from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("generation_logger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(LOG_DIR, f"generations-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"))
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


class MultipleChoiceSchema(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class BooleanSchema(BaseModel):
    answer: Literal["True", "False"]


def extract_schema(response:str, schema:BaseModel)->BaseModel:
    try:
        return schema(**json.loads(response))
    except:
        print(f"Error extracting schema from response: {response}")
        if schema == MultipleChoiceSchema:
            return MultipleChoiceSchema(answer=random.choice(["A", "B", "C", "D"]))
        elif schema == BooleanSchema:
            return BooleanSchema(answer=random.choice(["True", "False"]))
        else:
            raise ValueError(f"Unknown schema type: {schema}")


def inference_mode_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with torch.inference_mode() if self.inference else contextlib.nullcontext():
            return func(self, *args, **kwargs)
    return wrapper
        

class StructuredModelEvaluator:
    """Evaluate a model's ability to follow a structured output format, designed for smaller LLMs.
    
    This class implements a robust two-turn approach particularly suited for smaller language models
    (1B-7B parameters) which often struggle with consistent output formatting:
    
    1. First turn: Model thinks freely about the answer without constraints, allowing it to reason
       even when format constraints might interfere with its thinking
    2. Second turn: Model generates a schema-compliant response using the previous thoughts,    
       with strict format enforcement to overcome the tendency of smaller models to deviate
       from required formats
    
    The final response is validated against the provided schema (e.g. MultipleChoice, Boolean)
    with fallback options for when the model fails to follow the format.
    """
    def __init__(self, model, tokenizer, device="cuda", inference=True, do_sample=False, system_prompt="You are a helpful assistant.", adherence_prompt="Now answer the question in the required format."):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.inference, self.do_sample = inference, do_sample
        self.system_prompt, self.adherence_prompt = system_prompt, adherence_prompt

        self.model.to(self.device)

        if self.inference:
            self.model.eval()  # ensure dropout, etc. are disabled
            self.model.generation_config.use_cache = True

    @inference_mode_decorator
    def generate_with_schema(self, model_input, schema, max_new_tokens=20) -> str:
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
        return self.model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **(
                {"do_sample":True, "temperature":0.7, "top_p":0.95, "top_k":60}  # sampling
                if self.do_sample
                else
                {"do_sample":False, "temperature":None, "top_p":None, "top_k":None}  # greedy decoding
            ),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_function,
        )

    @inference_mode_decorator
    def generate_without_schema(self, model_input, max_new_tokens=20) -> str:
        return self.model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **(
                {"do_sample":True, "temperature":0.7, "top_p":0.95, "top_k":60}  # sampling
                if self.do_sample
                else
                {"do_sample":False, "temperature":None, "top_p":None, "top_k":None}  # greedy decoding
            ),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        )
    
    def process_messages(self, messages:list[list[dict]])->dict:
        formatted = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        return self.tokenizer(formatted, return_tensors="pt", padding=True, add_special_tokens=True, padding_side="left").to(self.device)

    def generate(self, prompts:str|list[str], schema:BaseModel, max_first_turn_tokens:int=200, max_second_turn_tokens:int=20)->list[str|BaseModel]:
        prompts = [prompts] if isinstance(prompts, str) else prompts
    
        # First turn - let model think freely
        messages = [[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": p}
        ] for p in prompts]
        model_input = self.process_messages(messages)
        thought_ids = self.generate_without_schema(model_input, max_new_tokens=max_first_turn_tokens)
        thoughts = self.tokenizer.batch_decode(thought_ids[:, len(model_input.input_ids[0]):], skip_special_tokens=True)
        
        # Second turn - enforce schema with context
        for m, t in zip(messages, thoughts):
            m.append({"role": "assistant", "content": t})
            m.append({"role": "user", "content": self.adherence_prompt})
        
        model_input = self.process_messages(messages)
        generated_ids = self.generate_with_schema(model_input, schema, max_new_tokens=max_second_turn_tokens)
        outputs = self.tokenizer.batch_decode(generated_ids[:, len(model_input.input_ids[0]):], skip_special_tokens=True)
        final_outputs = [extract_schema(output.strip(), schema) for output in outputs]

        for p, t, o in zip(prompts, thoughts, outputs):
            logger.info(json.dumps({
                "prompt": p,
                "thought": t,
                "output": o,
                "model": self.model.name_or_path,
                "system_prompt": self.system_prompt,
                "adherence_prompt": self.adherence_prompt,
            }, indent=4))

        return final_outputs


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.generation_config.pad_token_id = tokenizer.pad_token_id  # suppress warnings

    evaluator = StructuredModelEvaluator(model, tokenizer, inference=True, do_sample=True)

    print(f"\n{'='*20} Not batched {'='*20}\n")

    question = "What is the capital of France? A: Paris B: London C: Rome D: Madrid"
    print(question)
    print(evaluator.generate(
        question, 
        schema=MultipleChoiceSchema,
        max_first_turn_tokens=100,
        max_second_turn_tokens=10,
    ))

    question = "\nIs Paris the capital of France? True or False"
    print(question)
    print(evaluator.generate(
        question, 
        schema=BooleanSchema,
        max_first_turn_tokens=100,
        max_second_turn_tokens=10,
    ))

    print(f"\n{'='*20} Batched {'='*20}\n")
    
    questions = ["What is the capital of France? A: Paris B: London C: Rome D: Madrid", "What is the capital of Iceland? A: Oslo B: Reykjavik C: Stockholm D: Helsinki"]
    answers = evaluator.batch_generate(
        questions,
        schema=MultipleChoiceSchema,
        max_first_turn_tokens=100,
        max_second_turn_tokens=10,
    )    
    for q, a in zip(questions, answers):
        print(q, "\n", a, "\n")

    questions = ["Is Paris the capital of France? True or False", "Is Reykjavik the capital of Iceland? True or False"]
    answers = evaluator.batch_generate(
        questions,
        schema=BooleanSchema,
        max_first_turn_tokens=100,
        max_second_turn_tokens=10,
    )
    for q, a in zip(questions, answers):
        print(q, "\n", a, "\n")
