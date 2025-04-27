import os
import uuid
import json
import logging
import pathlib
import subprocess
import multiprocessing as mp
from typing import Dict, Any, List, Tuple

from trl.extras.vllm_client import VLLMClient   # your existing base‑class

from src.utils.git import get_head_commit_diff, handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)  # Force spawn method for multiprocessing 

class CodexAgent(VLLMClient):
    ...
    
# calls into the codex cli in full auto mode