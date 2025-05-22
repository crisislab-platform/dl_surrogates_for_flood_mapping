from datetime import datetime
import torch
import logging
import random
import string
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Add 6 random alphanumeric characters
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    # Include process ID for further uniqueness
    pid = os.getpid()
    return f"{timestamp}_{random_suffix}_{pid}"

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    return device
