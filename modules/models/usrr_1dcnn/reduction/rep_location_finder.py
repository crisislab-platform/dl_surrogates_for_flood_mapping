from modules.lib.constants import DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR, RUN_DIR
from modules.models.usrr_1dcnn.lib.base_functions import *
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray, gdal_transform, rc2coords, gdal_writeasc
from modules.models.usrr_1dcnn.reduction.rl_culster_finder import RLClusterFinder
from modules.utils.path_util import ensure_dir
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map

import logging
import numpy as np
import pandas as pd
import os
import torch
import psutil
import time
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Representaitve_Location_Finder")

model_name = "USRR_1DCNN_REDUCTION"

class RepLocation:
    def __init__(self, results_dir):
        self.work_dir = results_dir
        ensure_dir(self.work_dir)
        logger.info(f"Representative Location Finder initialized with work directory: {self.work_dir}")
        
    
    def spatial_sampling_new(self, run_id, dem_asc_file, max_inun_file, sampling_dist):
        potential_inun_arr = torch.from_numpy(gdal_asarray(max_inun_file)).cuda()
        dem_arr = torch.from_numpy(gdal_asarray(dem_asc_file)).cuda()
        
        inundation_mask = (potential_inun_arr > 0.3)
        inundated_cells_count = inundation_mask.sum().item()
        logger.info(f"Total inundated cells: {inundated_cells_count} out of {potential_inun_arr.shape[0] * potential_inun_arr.shape[1]}")
        
        pixel_width = 5
        num_cell_per_sample = int(sampling_dist / pixel_width)
        num_block_0 = (potential_inun_arr.shape[0] + num_cell_per_sample - 1) // num_cell_per_sample  # dir y (height)
        num_block_1 = (potential_inun_arr.shape[1] + num_cell_per_sample - 1) // num_cell_per_sample  # dir x (width)
        rl_tensor  = torch.zeros((potential_inun_arr.shape[0], potential_inun_arr.shape[1]), dtype=torch.float32)
        
        for i_0 in range(num_block_0):
            start_0 = i_0 * num_cell_per_sample
            end_0 = min(start_0 + num_cell_per_sample, potential_inun_arr.shape[0])
            for i_1 in range(num_block_1):
                start_1 = i_1 * num_cell_per_sample
                end_1 = min(start_1 + num_cell_per_sample, potential_inun_arr.shape[1])
                
                inundation_block = inundation_mask[start_0:end_0, start_1:end_1]
                if not torch.any(inundation_block):
                    logger.info(f"Block {i_0}, {i_1} has no inundation - skipping")
                    continue
                
    
                curr_dem_block_arr = dem_arr[start_0:end_0, start_1:end_1]
   
                masked_dem = curr_dem_block_arr.clone()
                masked_dem[~inundation_block] = float('inf')
                argmin_dem = torch.argmin(masked_dem.view(-1)).item()
                block_width = end_1 - start_1
                ri = argmin_dem // block_width
                ci = argmin_dem % block_width
                rl_tensor[start_0 + ri, start_1 + ci] = 1
                logger.info(f"Block {i_0}, {i_1} - Minimum DEM value at ({start_0 + ri}, {start_1 + ci})")
        
        # Save the rl_tensor as a .asc file
        output_asc_file = f'{self.work_dir}/rl_{sampling_dist}.asc'
        gdal_writeasc(output_asc_file, rl_tensor.cpu().numpy(), dem_asc_file)
        self.visualise_points(output_asc_file, dem_asc_file, f'{self.work_dir}/rl_{sampling_dist}.png')
        
        return output_asc_file
    
    def visualise_points(self, rl_asc_file, dem_asc_file, output_file):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from matplotlib import colorbar
        
        rl_arr = gdal_asarray(rl_asc_file)
    
        # Load DEM data
        dem_arr = gdal_asarray(dem_asc_file)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot DEM as background with terrain colormap
        dem_plot = plt.imshow(dem_arr, cmap='terrain', alpha=0.8)
        
        # Create colorbar for DEM elevation
        cbar = plt.colorbar(dem_plot)
        cbar.set_label('Elevation (m)')
        
        # Find representative locations (where rl_arr is 1)
        rl_points_y, rl_points_x = np.where(rl_arr == 1)
        
        # Plot representative locations as blue points
        plt.scatter(rl_points_x, rl_points_y, c='blue', s=30, marker='o', 
                   edgecolors='black', linewidths=0.5, label='Representative Locations')
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Add title and labels
        plt.xlabel('X (Column)')
        plt.ylabel('Y (Row)')
        
        plt.title('Representative Locations Overlayed on DEM')
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Representative locations visualized and saved to {output_file}")
    
def find_representative_locations_and_clusters(run_id, sampling_dist, n_clusters=10, random_state=42, n_init=10):
    
    process = psutil.Process()
    start_cpu_time = process.cpu_times().user + process.cpu_times().system
    start_time = time.time()

    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                                profile_memory=True) as prof:

        rl_file_path = reducer_func(run_id, sampling_dist, n_clusters, random_state, n_init)
    
    logger.info(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    analysis = profiler_analysis(prof.key_averages())
    logger.info(analysis)
        
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
            'max_cuda_memory': analysis.get('max_cuda_memory', 0),
            'total_cpu_time': cpu_time_used,
            'total_gpu_time': analysis.get('total_gpu_time', 0),
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
            'max_cuda_memory': analysis.get('max_cuda_memory', 0),
            'total_cpu_time': cpu_time_used,
            'total_gpu_time': analysis.get('total_gpu_time', 0),
            'cpu_time_used': cpu_time_used
        }
        df = pd.DataFrame([new_row])
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
    logger.info(f"Total CPU time used: {cpu_time_used} seconds")
    logger.info(f"Total wall time used: {wall_time} seconds")       
    logger.info(f"Estimated FLOPS: {format_flops(estimated_flops)}")
    logger.info("SDR reduction completed successfully.")
    
    return rl_file_path
    
    
def reducer_func(run_id, sampling_dist, n_clusters=10, random_state=42, n_init=10):
    work_dir = f"{OUTPUT_DIR}/rls"
    dem_asc_file = f"{SIMULATION_DATA_DIR}/Carlisle_5m.asc"
    simulation_dir = SIMULATION_DATA_DIR
    max_inunundation_file = f"{simulation_dir}/Run3-0094.wd"
    run_meta_data_file = f"{work_dir}/run_meta_data.csv"
    
    rep_loc = RepLocation(work_dir)
    rl_file_path = rep_loc.spatial_sampling_new(run_id, dem_asc_file, max_inunundation_file, sampling_dist=sampling_dist)
    
    # Create the cluster finder with GPU awareness
    
    cluster_finder = RLClusterFinder(work_dir, run_id, rl_file_path, sampling_dist, n_clusters=n_clusters,
                                    random_state=random_state, n_init=n_init, raster_temp=dem_asc_file)

    # Run clustering
    cluster_finder.run_clustering_new()
    
    # Save the meta data to csv files
    meta_data = {
        'run_id': run_id,
        'sampling_dist': sampling_dist,
        'n_clusters': n_clusters,
        'random_state': random_state,
        'n_init': n_init,
        'rl_file_path': rl_file_path
    }
    
    # Convert to DataFrame and save to CSV (append if file exists)
    df_new = pd.DataFrame([meta_data])
    
    if os.path.exists(run_meta_data_file):
        df_new.to_csv(run_meta_data_file, mode='a', header=False, index=False)
        logger.info(f"Appended run metadata to {run_meta_data_file}")
    else:
        # Create new file with header
        df_new.to_csv(run_meta_data_file, index=False)
        logger.info(f"Created new run metadata file at {run_meta_data_file}")

    logger.info(f"Representative locations and clusters found and saved")
    return rl_file_path

