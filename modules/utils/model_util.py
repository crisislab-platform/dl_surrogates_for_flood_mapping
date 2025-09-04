import os
import psutil
import logging
import torch
import rasterio
from typing import Dict
from modules.lib.constants import SIMULATION_DATA_DIR, RUN_DIR

# Assuming logger is from the logging module
logger = logging.getLogger(__name__)

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

def format_flops(flops):
        if (flops < 1e9):
            return f"{flops / 1e6:.2f} MFLOPS"
        else:
            return f"{flops / 1e9:.2f} GFLOPS"
    
def profiler_analysis(key_averages):
    try:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cuda_memory_values = torch.tensor([getattr(item, 'self_device_memory_usage', 0) for item in key_averages], 
                                            dtype=torch.float32, device=device)
        cpu_memory_values = torch.tensor([getattr(item, 'self_cpu_memory_usage', 0) for item in key_averages], 
                                        dtype=torch.float32, device=device)
        max_cuda_memory = torch.max(cuda_memory_values).item() if len(cuda_memory_values) > 0 else 0
        max_cpu_memory = torch.max(cpu_memory_values).item() if len(cpu_memory_values) > 0 else 0
        
        # Extract total memory usage on GPU
        total_cuda_memory = torch.sum(cuda_memory_values).item() if len(cuda_memory_values) > 0 else 0
        total_cpu_memory = torch.sum(cpu_memory_values).item() if len(cpu_memory_values) > 0 else 0
        
        # Extract total CPU and GPU time
        cpu_time_values = torch.tensor([getattr(item, 'cpu_time_total', 0) for item in key_averages], 
                                        dtype=torch.float32, device=device)
        gpu_time_values = torch.tensor([getattr(item, 'device_time_total', 0) for item in key_averages], 
                                        dtype=torch.float32, device=device)
        
        total_cpu_time = torch.sum(cpu_time_values).item() if len(cpu_time_values) > 0 else 0
        total_gpu_time = torch.sum(gpu_time_values).item() if len(gpu_time_values) > 0 else 0
        
        flops_values = torch.tensor([getattr(item, 'flops', 0) for item in key_averages],
                             dtype=torch.float32, device=device)
        flops = torch.sum(flops_values).item() if len(flops_values) > 0 else 0
        
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        # Log the results
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
            "total_gpu_time": total_gpu_time,
            "flops": flops
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
            "flops": 0,
            "error": str(e)
        }

def save_prediction_map(prediction_map, output_dir, idx, model_name):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving prediction maps to {output_dir}")

    # Get reference raster for metadata
    ref_file = os.path.join(SIMULATION_DATA_DIR, "Run1-0000.wd")
    with rasterio.open(ref_file) as src:
        height = src.height
        width = src.width
        profile = src.profile
        transform = src.transform
        crs = src.crs
    
        
    logger.info(f'Prediction shape: {prediction_map.shape}, Reshaping to dimensions: {height}x{width}')
    pred_reshaped = prediction_map.reshape(height, width)
    
    # Define output file path
    out_file = os.path.join(output_dir, f"map_{idx:04d}.wd")
    
    # Create a copy of the profile for the output file
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    # Write the reshaped prediction directly to a .wd file
    with rasterio.open(out_file, 'w', **out_profile) as dst:
        #convet to numpy if tensor
        if isinstance(pred_reshaped, torch.Tensor):
            pred_reshaped = pred_reshaped.cpu().numpy()
        dst.write(pred_reshaped.astype(rasterio.float32), 1)
    
    logger.info(f"Saved prediction map {idx} to {out_file}")
    return output_dir
