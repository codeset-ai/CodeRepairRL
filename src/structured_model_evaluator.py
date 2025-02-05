import json
import random
import contextlib
from typing import Literal
from functools import wraps


from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn


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
        # Use inference_mode if enabled, otherwise a no-op context.
        context = torch.inference_mode() if self.inference_only else contextlib.nullcontext()
        with context:
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
    def __init__(self, model, tokenizer, name, device="cuda", system_prompt="You are a helpful assistant.", inference_only=True, use_cache=True):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.device = device
        self.system_prompt = system_prompt
        self.inference_only = inference_only
        self.use_cache = use_cache

        self.model.to(self.device)
        self.model.eval()  # ensure dropout, etc. are disabled
        # Ensure the generation config is updated accordingly if necessary:
        self.model.generation_config.use_cache = use_cache

    @inference_mode_decorator
    def generate_with_schema(self, model_input, schema, max_new_tokens=20) -> str:
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
        return self.model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            use_cache=self.use_cache,
            do_sample=False, temperature=None, top_p=None, top_k=None,  # greedy decoding
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_function,
        )

    @inference_mode_decorator
    def generate_without_schema(self, model_input, max_new_tokens=20) -> str:
        return self.model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            use_cache=self.use_cache,
            do_sample=False, temperature=None, top_p=None, top_k=None,  # greedy decoding
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        )
    
    def process_messages(self, messages:list[dict]|list[list[dict]])->dict:
        """Convert message histories to model inputs. Handles both single and batch inputs."""
        if not isinstance(messages[0], list):
            messages = [messages]  # make it batch
        
        formatted = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        return self.tokenizer(formatted, return_tensors="pt", padding=True, add_special_tokens=True, padding_side="left").to(self.device)

    def generate(self, prompt:str, schema:BaseModel, max_first_turn_tokens:int=200, max_second_turn_tokens:int=10)->str:
        # First turn - let model think freely
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": prompt})
        model_input = self.process_messages(messages)
        thought_ids = self.generate_without_schema(model_input, max_new_tokens=max_first_turn_tokens)
        thoughts = self.tokenizer.decode(thought_ids[0, len(model_input.input_ids[0]):], skip_special_tokens=True)

        # Second turn - enforce schema with context
        messages.append({"role": "assistant", "content": thoughts})
        messages.append({"role": "user", "content": "Now answer the question in the required json format."})
        model_input = self.process_messages(messages)

        generated_ids = self.generate_with_schema(model_input, schema, max_new_tokens=max_second_turn_tokens)
        answer = self.tokenizer.decode(generated_ids[0, len(model_input.input_ids[0]):], skip_special_tokens=True)
        return extract_schema(answer.strip(), schema)


    def batch_generate(self, prompts:list[str], schema:BaseModel, max_first_turn_tokens:int=200, max_second_turn_tokens:int=10)->list[str|BaseModel]:
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
            m.append({"role": "user", "content": "Now answer the question in the required format."})
        
        model_input = self.process_messages(messages)
        generated_ids = self.generate_with_schema(model_input, schema, max_new_tokens=max_second_turn_tokens)
        outputs = self.tokenizer.batch_decode(generated_ids[:, len(model_input.input_ids[0]):], skip_special_tokens=True)
        return [extract_schema(output.strip(), schema) for output in outputs]

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.generation_config.pad_token_id = tokenizer.pad_token_id  # suppress warnings

    evaluator = StructuredModelEvaluator(model, tokenizer, model_name)

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
        print(q)
        print(a, "\n")

    questions = ["Is Paris the capital of France? True or False", "Is Reykjavik the capital of Iceland? True or False"]
    answers = evaluator.batch_generate(
        questions,
        schema=BooleanSchema,
        max_first_turn_tokens=100,
        max_second_turn_tokens=10,
    )
    for q, a in zip(questions, answers):
        print(q)
        print(a, "\n")
