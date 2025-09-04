from modules.models.srr_lstm.srr.srr_reconstruction import SDRReconstructor
from modules.models.srr_lstm.srr.srr_reduction import SDRReducer
import logging
import psutil
import time
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map
from modules.lib.constants import RUN_DIR
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SRR_Reduction")

model_name = "SRR_LSTM_REDUCTION"

def findRLS(run_id):
    process = psutil.Process()
    start_cpu_time = process.cpu_times().user + process.cpu_times().system
    start_time = time.time()

    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    reducer = SDRReducer()
    reducer.sdr_searching()
    reducer.sdr_searching_mcl()
    reducer.sdr_rl()
    
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    memory_used = mem_after - mem_before
    logger.info(f"Memory used during SDR reduction: {memory_used:.2f} MB")
    
    end_time = time.time()
    end_cpu_time = process.cpu_times().user + process.cpu_times().system
    cpu_time_used = end_cpu_time - start_cpu_time
    wall_time = end_time - start_time
    
    # Estimate FLOPS based on CPU frequency and utilization
    cpu_freq = psutil.cpu_freq().current * 1e6  # Convert MHz to Hz
    cpu_count = psutil.cpu_count(logical=True)
    utilization = cpu_time_used / wall_time

    estimated_flops = cpu_freq * cpu_count * utilization * wall_time
    print(f"Estimated FLOPS: {estimated_flops:,.0f}")
    
    logger.info(f"Memory profiling completed. Results saved to lstm_model_memory_usage.txt")
    
    # Save the results to a csv file in the output directory
    # Create path if it does not exist
    output_dir = os.path.join(RUN_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"reduction_metrics.csv")
    
    # Append the results to the csv file
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        new_row = {
            'run_id': run_id,
            'reduction_time': wall_time,
            'flops': estimated_flops,
            'max_cpu_memory':  memory_used,
            'max_cuda_memory': 0,
            'total_cpu_time': cpu_time_used,
            'total_gpu_time': 0,
            'cpu_time_used': cpu_time_used
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(output_file, index=False)  
        logger.info(f"Results saved to {output_file}")
        
    else:
        new_row = {
            'run_id': run_id,
            'reduction_time': wall_time,
            'flops': estimated_flops,
            'max_cpu_memory':  memory_used,
            'max_cuda_memory': 0,
            'total_cpu_time': cpu_time_used,
            'total_gpu_time': 0,
            'cpu_time_used': cpu_time_used
        }
        df = pd.DataFrame([new_row])
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
    logger.info(f"Total CPU time used: {cpu_time_used} seconds")
    logger.info(f"Total wall time used: {wall_time} seconds")       
    logger.info(f"Estimated FLOPS: {format_flops(estimated_flops)}")
    logger.info("SDR reduction completed successfully.")
    