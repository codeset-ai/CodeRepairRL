# Just an idea to keep track of (this is just pseudocode)
# we dont specifically care about search, but solving this "simpler" problem might shed light on how to do function calling / running test suites on patches

# Can we train a model to use search (or other tools) like this
#  1. Setup function calling, e.g. <search query="..."> Dump the search results here </search>
#  2. Train the model with rewards for correct answers, but penalize if search lead to a wrong answer (actual negative reward)
#      - The hope being that search will lead to more correct answers, thus training the model to use search
#  3. Also include some format rewards, for <think>, <answer> and <search> tags
#      - Weigh the search penalty to be higher than the format reward
#      - Perhaps always penalize <search> tags and hope that the signal that correct answers are more frequent via search outweighs the penalty
#  4. Profit???

import re

import torch
from trl import GRPOTrainer
from transformers import StoppingCriteria, AutoTokenizer
from typing import Optional
from duckduckgo_search import DDGS


def search_web(query: str, max_results: Optional[int] = 3) -> list[str]:
    """Perform a web search using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 3)
        
    Returns:
        List of search result snippets
    """
    with DDGS() as ddgs:
        results = []
        for i, r in enumerate(ddgs.text(query, max_results=max_results)):
            results.append(f"{i+1}: {r['body']}")
    return results


class SearchStoppingCriteria(StoppingCriteria): # probably not a StoppingCriteria, should be a LogitsProcessor since we use vllm
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

        self.search_start_token = "<search query="
        self.search_end_token = ">"
        
        self.started = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        if self.search_start_token in self.tokenizer.decode(input_ids[0]) and not self.started:
            self.started = True
            return False
        elif self.search_end_token in self.tokenizer.decode(input_ids[0]) and self.started:
            self.started = False
            return True
        else:
            return False


class SearchGRPOTrainer(GRPOTrainer):
    
    # one problem is how we do this in a batched fashion (MAJOR PROBLEM)
    # probably best to:
    # 1. stop all generations in the batch
    # 2. perform the search
    # 3. append the search results to that specific batch item
    # 4. clear away left padding if possible
    # 5. left pad the batch to match
    # 6. continue generating

    # do this in a loop with e.g. max 3 searches
    def generate_with_search(self, **kwargs):
        # generate until we see a <search query="..."> tag
        text = self.vllm.generate(
            **kwargs,
            stopping_criteria=[SearchStoppingCriteria(self.model.tokenizer)]
        )

        # grab the query
        pattern = re.compile(r"<search query=\"(.*?)\">")
        match = pattern.search(text)
        if match:
            query = match.group(1)
            print(f"Query: {query}")
            results = search_web(query)
            print(f"Results: {results}")

        text += "\n" + "\n".join(results) + "\n" + "</search>"

        text = self.model.generate(
            text,
            stopping_criteria=[SearchStoppingCriteria(self.model.tokenizer)]
        )

        return text
            