from dataclasses import dataclass
import os
import json
import gc
import logging
import time
import torch
import psutil
import os
import gc
from typing import Dict, Tuple


logger = logging.getLogger("Model")

# Unified Model configuration
@dataclass
class ModelConfig:
    """Unified configuration class for model parameters and training"""
    model_name: str
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    epochs: int = 10
    patience: int = 2
    run_id: str = None
    run_dir: str = None
    args: dict = None

class ModelWrapper:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.train_dataset = None
        self.val_dataset = None
        self.x_test = None
        self.y_test = None
        self.grid_height = None
        self.grid_width = None
        self.train_steps = None
        self.val_steps = None
        self.model = None
        
    def init_model(self) -> bool:
        pass
    
    def create_dataset(self):
        pass
    
    def train(self, run_dir: str):
        pass
    
    def predict(self, run_id: str = None, model_file: str = None):
        pass
    
    def validate_model(self):
        pass

    def save_model_checkpoint(self, run_id, run_dir, model, config):
        pass
    
    # Add these helper functions to your class
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage for CPU and GPU."""
        memory_stats = {
            "cpu_percent": psutil.Process(os.getpid()).memory_percent(),
            "cpu_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),  # Convert to MB
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            memory_stats.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024)
            })
        
        return memory_stats

    def log_memory_stats(self, phase="training", epoch=None, batch=None):
        """Log memory statistics with appropriate context."""
        stats = self.get_memory_usage()
        
        # Format the context
        context = phase
        if epoch is not None:
            context += f" epoch {epoch}"
        if batch is not None:
            context += f" batch {batch}"
        
        # Log the stats
        mem_msg = f"Memory usage ({context}): CPU: {stats['cpu_mb']:.2f}MB ({stats['cpu_percent']:.2f}%)"
        if torch.cuda.is_available():
            mem_msg += f", GPU allocated: {stats['gpu_allocated_mb']:.2f}MB, reserved: {stats['gpu_reserved_mb']:.2f}MB"
        
        logger.info(mem_msg)
        return stats