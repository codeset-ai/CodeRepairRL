import torch
import subprocess


def resolve_git_commit_hash(git_commit_hash=None):
    """To know in which commit the code was run"""
    if git_commit_hash is not None:
        return git_commit_hash
    
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except: return "unknown"