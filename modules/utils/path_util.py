import os

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path