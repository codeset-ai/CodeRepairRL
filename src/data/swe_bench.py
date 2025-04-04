import logging
from typing import Tuple, Optional

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

from data.code_mono_repair import create_mono_repair_dataset
from src.utils.git import clone_repo_at_commit, handle_to_url, clean_repo_dir

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
    from tqdm import tqdm
    from datasets import load_dataset

    swe_ds = load_dataset("princeton-nlp/SWE-bench_Lite")["dev"]

    url = handle_to_url(swe_ds[0]["repo"])
    repo_path = clone_repo_at_commit(url, swe_ds[0]["base_commit"])
    print(repo_path)
    clean_repo_dir(repo_path)



    # repo_paths = []
    # for item in tqdm(swe_ds):
    #     repo_url = handle_to_url(item["repo"])
    #     repo_path = clone_repo_at_commit(repo_url, item["base_commit"])
    #     repo_paths.append(repo_path)
