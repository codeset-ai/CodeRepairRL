import git
import tempfile
import subprocess


def handle_to_url(repo_handle):
    return f"https://github.com/{repo_handle}.git"

def clone_repo_at_commit(repo_url, commit_id, target_dir=None):
    """
    Fast clone method with no git history. Initializes an empty repo, adds the remote then fetches the specific commit and checks it out.
    
    Args:
        repo_url: Repository URL
        commit_id: Commit hash to checkout
        target_dir: Optional target directory. If None, creates a temporary directory.
        
    Returns:
        Path to the cloned repository
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp()
    
    
    repo = git.Repo.init(target_dir)
    origin = repo.create_remote('origin', repo_url)
    origin.fetch(commit_id, depth=1)
    repo.git.checkout(commit_id)

    return target_dir

def resolve_git_commit_hash(git_commit_hash=None):
    """To know in which commit the code was run"""
    if git_commit_hash is not None:
        return git_commit_hash
    
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except: return "unknown"  
