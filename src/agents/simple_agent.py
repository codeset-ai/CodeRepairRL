import logging
import multiprocessing as mp
from typing import Dict, Any, List, Tuple

from trl.extras.vllm_client import AsyncVLLMClient


logger = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)  # Force spawn method for multiprocessing


