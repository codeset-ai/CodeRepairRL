import re

import wandb

from src.utils.logging import build_html_table
from src.utils.diff import SearchReplaceDiff, UnifiedDiff


def extract_xml_answer(text:str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return "N/A"


# FYI: rewards are calculated as callbacks which receive the keywords prompts, completions and all the columns of your dataset as kwargs

##################################################################################################################  
# Reasoning style response reward functions
##################################################################################################################

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

def partial_reasoning_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function to guide models to use a specific format for reasoning. Does not account for ordering and placement of tags."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def strict_reasoning_format_reward_func(completions, **kwargs) -> list[float]:
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

##################################################################################################################
# Detection specific reward functions
##################################################################################################################

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

##################################################################################################################
# Repair specific reward functions
# when create_repair_dataset is called, the diff_type becomes a column and is therefore present in the kwargs
##################################################################################################################

def diff_format_reward_func(completions, diff_type, **kwargs) -> list[float]:
    """Reward function that checks the quality of the extracted diff format between 0.0 and 1.0."""
    diff_cls = SearchReplaceDiff if diff_type == "search_replace" else UnifiedDiff
    
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_xml_answer(c) for c in contents]  # extract contents of <answer> tags
    diffs_batch = [diff_cls.extract_all(diff_str) for diff_str in answers]  # batch of a list of diffs (for potentially multiple files)
    
    return [
        sum(diff.validate_quality() for diff in diffs)/(len(diffs) or 1)
        for diffs in diffs_batch
    ] 

def diff_similarity_reward_func(prompts, completions, diffs, diff_type, **kwargs) -> list[float]:
    """Reward function that sequence matches the reference and generated diffs."""
    diff_cls = SearchReplaceDiff if diff_type == "search_replace" else UnifiedDiff 
    
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_xml_answer(c) for c in contents]  # extract contents of <answer> tags
    generated_diffs_batch = [diff_cls.extract_all(diff_str) for diff_str in answers]
    reference_diffs_batch = [diff_cls.extract_all(diff_str) for diff_str in diffs]  # TODO: support having multiple reference diffs

    #########################################################
    # Nasty hack, GRPOTrainer offers no other way to create callbacks with the completions
    html_rows = []
    for prompt_item, response, ext, ans in zip(prompts, contents, generated_diffs_batch, reference_diffs_batch):
        prompt_text = prompt_item[-1]['content'] if prompt_item else ""
        gen_diff_str = "\n\n".join([diff.to_string() for diff in ext])
        ref_diff_str = "\n\n".join([diff.to_string() for diff in ans])
        html_rows.append((prompt_text, response, gen_diff_str, ref_diff_str))
    html_table = build_html_table(html_rows)
    wandb.log({"eval_table": wandb.Html(html_table)})
    #########################################################
    
    # TODO: ugly, support multi file diffs better
    rewards = []
    for ref_diffs, gen_diffs in zip(reference_diffs_batch, generated_diffs_batch):
        assert len(ref_diffs) == len(gen_diffs), "Number of reference diffs and generated diffs must be the same"
        per_file_similarity = []
        for ref_diff, gen_diff in zip(ref_diffs, gen_diffs):
            per_file_similarity.append(ref_diff.similarity(gen_diff))
        rewards.append(sum(per_file_similarity)/(len(per_file_similarity) or 1))

    return rewards