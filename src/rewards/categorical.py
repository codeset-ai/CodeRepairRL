from src.utils.extract import extract_xml_answer

# Detection specific reward functions

def categorical_correctness_reward_func(completions, answers, **kwargs) -> list[float]:
    """Reward function that checks if the extracter answer matches the ground truth answer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    return [1.0 if ext == a else 0.0 for ext, a in zip(extracted_responses, answers)]