# hello_sweagent.py
import os
import json
from pathlib import Path

from sweagent.run.run_single import run_from_config, RunSingleConfig


# ---------- helpers you already have -----------------------------------------
from src.data import get_swe_gym_repo_repair_dataset          # provides HF dataset
from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir                # clones and returns path
# ------------------------------------------------------------------------------

# Directory that will NEVER live inside any git repo
BASE_RUN_DIR = Path.home() / "sweagent_runs"
BASE_RUN_DIR.mkdir(parents=True, exist_ok=True)

def run_one(example, run_id="hello-world"):
    """
    example = {
        'repo'           : 'org/name',
        'base_commit'    : 'abcdef0',
        'problem_statement' : '…'
    }
    """
    repo_path = clone_repo_at_commit(handle_to_url(example["repo"]), example["base_commit"])

    # 2) pick a dedicated output folder
    out_dir = BASE_RUN_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) build a *schema-correct* config
    cfg = RunSingleConfig(
        output_dir=str(out_dir),            # replaces old logging.output_dir
        agent=dict(
            model=dict(
                # Prepend provider to model name so LiteLLM knows which backend to use
                name="anthropic/claude-3.5-haiku",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_input_tokens=8_000,
                max_output_tokens=4_096,
                # Avoid Anthropic empty system message error
                convert_system_to_user=False,
            ),
            templates=dict(
                system_template="You are a helpful software engineering agent."
            ),
            tools=dict(
                parse_function=dict(type="thought_action")
            )
        ),
        env=dict(
            deployment=dict(
                type="local",               # <- discriminator discriminator field
            ),
            repo=dict(
                type="preexisting",
                repo_name=str(repo_path).lstrip("/"),
                base_commit=example["base_commit"],
            ),
        ),
        problem_statement=dict(
            text=example["problem_statement"],
        ),
    )

    # 4) run synchronously
    run_from_config(cfg)

    # 5) read the last assistant message
    traj_file = max(out_dir.glob("trajectory_*.jsonl"), key=lambda p: p.stat().st_mtime)
    with traj_file.open() as fp:
        for line in fp:
            evt = json.loads(line)
            if evt["type"] == "assistant_message":
                print("\n=== assistant said ===\n", evt["content"])
                break

    # 6) cleanup cloned repo
    clean_repo_dir(repo_path)


if __name__ == "__main__":
    # one toy sample from SWE-Gym
    ds = get_swe_gym_repo_repair_dataset().select(range(1))
    ex = dict(ds[0])      # repo, base_commit, problem_statement

    run_one(ex, run_id="hello-world")



sweagent run \
  --agent.model.name=claude-3-5-sonnet-20241022 \
  --agent.model.per_instance_cost_limit=2.00 \
  --env.deployment.type=local \
  --env.repo.github_url=https://github.com/SWE-agent/test-repo \
  --problem_statement.github_url=https://github.com/SWE-agent/test-repo/issues/1
