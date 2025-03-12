import logging
from typing import Tuple, Optional

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

from src.data.code_repair import create_repair_dataset


logger = logging.getLogger(__name__)


def get_swe_bench_dataset(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_prompt_length: int = 512,
    system_prompt: Optional[str] = None
) -> Tuple[Dataset, int]:
    """
    """
    pass


if __name__ == "__main__":
    import requests
    from datasets import DatasetDict

    # Raw GitHub content URL template with commit ID
    RAW_GITHUB_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"

    swe_bench = load_dataset("princeton-nlp/SWE-bench_Lite")

    for example in swe_bench:
        ...


    swe_bench_dict = DatasetDict({
        "train": swe_bench["train"].select(range(100)),
        "test": swe_bench["test"].select(range(100)),
    })

    swe_bench_dict.push_to_hub("ASSERT-KTH/SWE-bench_Lite_repoprompt")
