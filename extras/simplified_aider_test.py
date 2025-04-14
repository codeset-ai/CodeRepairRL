import os, sys, json
from pathlib import Path

# Import Aider components
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.git import clone_repo_at_commit, handle_to_url, clean_repo_dir, get_head_commit_diff

# Load the dataset
ds = load_dataset("princeton-nlp/SWE-bench_Lite")["dev"].shuffle(seed=42)

url = handle_to_url(ds[0]["repo"])
temp_folder = clone_repo_at_commit(url, ds[0]["base_commit"])

try:
    # Change to the repo's root directory so Aider can compute the repo-map
    original_dir = os.getcwd()
    os.chdir(temp_folder)

    coder = Coder.create(
        main_model = Model("haiku"),
        io = InputOutput(yes=True),
        suggest_shell_commands = False,
    )
    sub_result = coder.run(ds[0]["problem_statement"])
    messsages = coder.format_chat_chunks().all_messages()
    diff = get_head_commit_diff(temp_folder)
finally:
    clean_repo_dir(temp_folder)
    os.chdir(original_dir)
    

with open("messages.json", "w") as f:
    json.dump(messsages, f, indent=4)
    
with open("diff.txt", "w") as f:
    f.write(diff)
    