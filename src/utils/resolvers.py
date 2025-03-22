import torch
import subprocess


def resolve_git_commit_hash(git_commit_hash=None):
    """To know in which commit the code was run"""
    if git_commit_hash is not None:
        return git_commit_hash
    
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except: return "unknown"

def resolve_bf16(use_bf16=None):
    """Resolves whether to use BF16 based on GPU architecture."""
    if use_bf16 is not None:
        return use_bf16
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

def resolve_fp16(use_fp16=None):
    """Resolves whether to use FP16 based on GPU architecture."""
    if use_fp16 is not None:
        return use_fp16
    # Use FP16 on Pascal+ GPUs (SM 6.0+) only if BF16 is not supported
    bf16 = resolve_bf16()
    fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 6 and torch.cuda.get_device_capability(0)[0] < 8 
    return fp16 and not bf16  # if bf16 is supported, we don't use fp16