import re


def extract_xml_answer(text:str) -> str:
    """
    Extract content between <answer> tags from model outputs.
    
    Some models like Qwen-2.5-Coder and Deepseek distilled reasoning models
    are trained to be token efficient and often omit <think> and </answer> tags.
    (from a design standpoint <think> can always be skipped since it is the very first token every time)

    We attempt to extract the answer tag, and if nothing can be extracted we return "N/A"
    """
    if "<answer>" in text and "</answer>" not in text:  # if the closing tag is missing we return everything after the opening tag
        return text.split("<answer>", 1)[1].strip()
    
    if "<answer>" in text and "</answer>" in text:  # if both tags are present we return the content between them
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    
    return "N/A" # if no tags are present we return "N/A"

def extract_markdown_block(response: str) -> str:
    """
    Extract the first code block from a markdown response.
    
    If no code block is found we return the response itself.
    """
    match = re.search(r"```(?:.*?)\n(.*?)```", response, re.DOTALL)
    return match.group(1) if match else response