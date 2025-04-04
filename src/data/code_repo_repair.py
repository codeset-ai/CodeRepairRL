# Prepares a dataset of entire code repositories

# (OR we provide it with a list of git handles and commit ids, and it will clone the repos etc.)
# Required data to be a repo repair dataset:
# - repo_folder: a path to the folder containing the repo (we have cloned it to that location and e.g. initialized the Aider cache by instantiating the agent)
# - issue_description: a description of the issue
#   - could be e.g. a PR issue statement, or a list of failing tests if available
# - golden_patch: a patch that fixes the issue

# Would be nice to have in the case of actually executing the tests, but for now we focus on the golden patch and diff matching
# - Optional[fail_to_pass]: a list of tests which are failing
# - Optional[pass_to_pass]: other tests which were working before and also need to work after the patch

# Doesn't need a SYSTEM_PROMPT, since that is offloaded to our chosen coding agent.

from datasets import Dataset

# Move the rs
def create_repo_repair_dataset() -> Dataset:
    ...
