import logging
from typing import List, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


def create_repo_repair_dataset(
    repo_urls: List[str],
    repo_commit_hashes: List[str],
    search_replace_patches: List[str],
    descriptions: Optional[List[str]] = None,
) -> Dataset:
    """
    Create a dataset for entire repository repair tasks.
    
    Args:
        repo_urls: List of repository URLs
        repo_commit_hashes: List of commit hashes at which the issues occurred
        search_replace_patches: List of search/replace patches that fix the issues
        descriptions: Optional list of issue descriptions
        
    Returns:
        The processed dataset
    """
    assert len(repo_urls) == len(repo_commit_hashes) == len(search_replace_patches) == len(descriptions), "repo_urls, repo_commit_hashes, search_replace_patches, and descriptions must have the same length"
    
    # Create dataset items
    data_items = []
    for repo_url, commit_hash, patch, desc in zip(repo_urls, repo_commit_hashes, search_replace_patches, descriptions):
        # Create dataset item
        item = {
            "repo_url": repo_url,
            "repo_commit_hash": commit_hash,
            "patch": patch,
            "description": desc,
        }
        data_items.append(item)
        
    
    repair_data = Dataset.from_list(data_items)   
     
    repair_data = repair_data.shuffle(seed=42)
    
    return repair_data
