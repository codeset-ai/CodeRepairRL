import re


def extract_xml_answer(text:str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return "N/A"

def extract_markdown_block(response: str) -> str:
    """Extract the first code block from a markdown response, or return the response itself if no code block is found."""
    match = re.search(r"```(?:.*?)\n(.*?)```", response, re.DOTALL)
    return match.group(1) if match else response