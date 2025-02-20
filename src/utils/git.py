import subprocess

def get_commit_hash():
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except: "unknown"