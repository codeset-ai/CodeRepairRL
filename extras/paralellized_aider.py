import os, sys
from pathlib import Path
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr

# Install in new environment, reqs don't work since trl is fixed to main on my fork
# This will be a requirement in the other repo, so it won't be an issue there
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.git import clone_repo_at_commit, handle_to_url


def process_item(item):
    """Core processing logic without output redirection."""
    url = handle_to_url(item["repo"])
    temp_folder = clone_repo_at_commit(url, item["base_commit"])

    try:
        # Change to the repo's root directory so Aider can compute the repo-map
        original_dir = os.getcwd()
        os.chdir(temp_folder)
        
        # Redirect Aider's terminal output to the void
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            coder = Coder.create(
                main_model = Model("haiku"),
                io = InputOutput(yes=True)
            )
            sub_result = coder.run("Give me a concise, high-level summary of what this repository does.")

        print(sub_result)
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    ds = load_dataset("princeton-nlp/SWE-bench_Lite")["dev"].select(range(2))
    with mp.Pool(2) as pool:
        pool.map(process_item, ds)
    