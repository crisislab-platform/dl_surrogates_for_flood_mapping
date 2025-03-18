import os

PROJECT_ROOT = "/home/91/23016891/projects/carlisle"
RUN_DIR = f"{PROJECT_ROOT}/runs"
DATA_DIR = f"{PROJECT_ROOT}/carlisle-data"
OUTPUT_DIR = f"{PROJECT_ROOT}/out"

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path