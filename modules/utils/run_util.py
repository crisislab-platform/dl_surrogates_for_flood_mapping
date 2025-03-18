from datetime import datetime
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    return device
