import re

import wandb

from src.utils.logging import build_html_table
from src.utils.diff import SearchReplaceDiff, UnifiedDiff


def extract_xml_answer(text:str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return "N/A"

#########################################################
# Reasoning style response reward functions
#########################################################

def count_xml(text:str)->float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count -= len(text.split("<think>\n")[0])*0.001  # penalize slightly for thinking before "think"
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]))*0.001  # penalize slightly for answering after "answer"
    return max(count, 0.0)

def partial_reasoning_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function to guide models to use a specific format for reasoning. Does not account for ordering and placement of tags."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def strict_reasoning_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function for strict adherence to our specific reasoning format."""
    pattern = (
        r"^<think>\n"
        r"([\s\S]+?)\n"
        r"</think>\n"
        r"<answer>\n"
        r"([^\n]+)\n"
        r"</answer>\s*$"
    )
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

#########################################################
# Detection specific reward functions
#########################################################

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the extracter answer matches the ground truth answer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    #########################################################
    # Nasty hack, GRPOTrainer offers no other way to create callbacks with the completions
    html_rows = []
    for prompt_item, response, ext, ans in zip(prompts, responses, extracted_responses, answer):
         prompt_text = prompt_item[-1]['content'] if prompt_item else ""
         html_rows.append((prompt_text, response, ext, ans))
    html_table = build_html_table(html_rows)
    wandb.log({"eval_table": wandb.Html(html_table)})
    #########################################################

    return [1.0 if ext == a else 0.0 for ext, a in zip(extracted_responses, answer)]

#########################################################
# Repair specific reward functions
# call them with functool.partial to set the diff_type
#########################################################

def diff_format_reward_func(prompts, completions, diff_type="search_replace", **kwargs) -> list[float]:
    """Reward function that checks the quality of the extracted diff format between 0.0 and 1.0."""
    contents = [completion[0]["content"] for completion in completions]
    diff_cls = SearchReplaceDiff if diff_type == "search_replace" else UnifiedDiff
    diffs = [diff_cls.extract_from_llm_response(c) for c in contents]  # attempts to extract diffs
    return [diff.validate_quality() for diff in diffs] 

def diff_similarity_reward_func(prompts, completions, reference, diff_type="search_replace", **kwargs) -> list[float]:
    """Reward function that sequence matches the reference and generated diffs."""
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_xml_answer(c) for c in contents]
    diff_cls = SearchReplaceDiff if diff_type == "search_replace" else UnifiedDiff 
    generated_diffs = [diff_cls.extract_from_llm_response(a) for a in answers]
    reference_diffs = [diff_cls.from_string(ref) for ref in reference]  # TODO: support having multiple reference diffs

    #########################################################
    # Nasty hack, GRPOTrainer offers no other way to create callbacks with the completions
    html_rows = []
    for prompt_item, response, ext, ans in zip(prompts, contents, generated_diffs, reference_diffs):
        prompt_text = prompt_item[-1]['content'] if prompt_item else ""
        html_rows.append((prompt_text, response, ext, ans))
    html_table = build_html_table(html_rows)
    wandb.log({"eval_table": wandb.Html(html_table)})
    #########################################################

    return [ref_diff.similarity(gen_diff) for ref_diff, gen_diff in zip(reference_diffs, generated_diffs)]