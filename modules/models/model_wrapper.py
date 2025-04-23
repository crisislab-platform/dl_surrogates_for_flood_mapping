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
from torch.profiler import profile, ProfilerActivity

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
    
    def nse_fn(self, observed, predicted):
        observed_mean = torch.mean(observed)
        numerator = torch.sum((observed - predicted) ** 2)
        denominator = torch.sum((observed - observed_mean) ** 2)
        nse = 1 - (numerator / denominator)
        nse = nse.item()
        return nse 
    
    def format_flops(self, flops):
        """Convert FLOPS to a human-readable string."""
        if (flops < 1e9):
            return f"{flops / 1e6:.2f} MFLOPS"
        else:
            return f"{flops / 1e9:.2f} GFLOPS"
    
    def profiler_analysis(self, key_averages):
        """
        Analyze profiling results from PyTorch profiler.
        
        Args:
            key_averages: The key_averages object from PyTorch profiler
            
        Returns:
            Dictionary containing memory and time metrics
        """
        try:
            # Get the correct attribute names
            cuda_memory_values = [getattr(item, 'self_device_memory_usage', 0) for item in key_averages]
            cpu_memory_values = [getattr(item, 'self_cpu_memory_usage', 0) for item in key_averages]
            
            # Extract maximum memory usage
            max_cuda_memory = max(cuda_memory_values) if cuda_memory_values else 0
            max_cpu_memory = max(cpu_memory_values) if cpu_memory_values else 0
            
            # Extract total memory usage
            total_cuda_memory = sum(cuda_memory_values) if cuda_memory_values else 0
            total_cpu_memory = sum(cpu_memory_values) if cpu_memory_values else 0
            
            # Extract total CPU and GPU time
            cpu_time_values = [getattr(item, 'cpu_time_total', 0) for item in key_averages]
            gpu_time_values = [getattr(item, 'device_time_total', 0) for item in key_averages]
            
            total_cpu_time = sum(cpu_time_values) if cpu_time_values else 0
            total_gpu_time = sum(gpu_time_values) if gpu_time_values else 0
            
            # Log the results
            # Memory usage is in bytes
            logging.info(f"Maximum CUDA memory usage: {max_cuda_memory / (1024 ** 2):.2f} MB")
            logging.info(f"Maximum CPU memory usage: {max_cpu_memory / (1024 ** 2):.2f} MB")
            logging.info(f"Total CUDA memory usage: {total_cuda_memory / (1024 ** 2):.2f} MB")
            logging.info(f"Total CPU memory usage: {total_cpu_memory / (1024 ** 2):.2f} MB")
            logging.info(f"Total CPU time: {total_cpu_time / 1e6:.2f} ms")
            logging.info(f"Total GPU time: {total_gpu_time / 1e6:.2f} ms")
            
            return {
                "max_cuda_memory": max_cuda_memory,
                "max_cpu_memory": max_cpu_memory,
                "total_cuda_memory": total_cuda_memory,
                "total_cpu_memory": total_cpu_memory,
                "total_cpu_time": total_cpu_time,
                "total_gpu_time": total_gpu_time
            }
        except Exception as e:
            logging.error(f"Error in profiler analysis: {e}")
            return {
                "max_cuda_memory": 0,
                "max_cpu_memory": 0,
                "total_cuda_memory": 0,
                "total_cpu_memory": 0,
                "total_cpu_time": 0,
                "total_gpu_time": 0,
                "error": str(e)
            }