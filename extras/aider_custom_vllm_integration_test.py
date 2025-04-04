# We've modified trl's vllm-serve source code emulate OpenAI API compatability

import os, sys, time, subprocess
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Install in new environment, reqs don't work since trl is fixed to main on my fork
# This will be a requirement in the other repo, so it won't be an issue there
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.git import clone_repo_at_commit, handle_to_url, clean_repo_dir


def test_integration(repo_url: str, commit_hash: str, problem_statement: str):
    try:
        temp_folder = clone_repo_at_commit(repo_url, commit_hash)
        original_dir = os.getcwd()
        os.chdir(temp_folder)
        
        os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
        os.environ["OPENAI_API_KEY"] = "dummy-key"
        
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            coder = Coder.create(
                main_model=Model("gpt4o-mini"),  # just a placeholder
                io=InputOutput(yes=True)
            )
            partial_result = coder.run(problem_statement)
        
        print(partial_result) 
            
    finally:
        os.chdir(original_dir)
        clean_repo_dir(temp_folder)


if __name__ == "__main__":
    ds = load_dataset("princeton-nlp/SWE-bench_Lite")["dev"]
    repo_url = handle_to_url(ds[0]["repo"])
    commit_hash = ds[0]["base_commit"]
    
    # Start vLLM server
    print("Starting vLLM server...")
    server_log = open("vllm_server.log", "w")
    server = subprocess.Popen(
        ["trl", "vllm-serve", "--model", "Qwen/Qwen2.5-Coder-7B-Instruct", "--port", "8000"],
        stdout=server_log, stderr=server_log
    )
    time.sleep(30)  # Wait for server to start
    
    test_integration(repo_url, commit_hash) 
    
    server.terminate()
    server_log.close() 