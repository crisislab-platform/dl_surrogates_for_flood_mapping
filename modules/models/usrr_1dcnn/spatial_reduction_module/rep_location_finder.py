from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.base_functions import *
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_asarray, gdal_transform, rc2coords
from modules.models.usrr_1dcnn.spatial_reduction_module.rl_culster_finder import RLClusterFinder

import logging
import numpy as np
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Representaitve_Location_Finder")

class RepLocation:
    def __init__(self, results_dir):
        self.work_dir = results_dir
        logger.info(f"Representative Location Finder initialized with work directory: {self.work_dir}")
        
    def spatial_sampling(self, run_id, dem_asc_file, max_inun_file, sampling_dist):
        """ Block sampling of representative locations from the IWL map.
            Suitable for cases containing multiple mainstreams and complex IWL conditions. 
            Only includes blocks with inundation. """
        rl_coords_ls = []
        potential_inun_arr = gdal_asarray(max_inun_file)
        dem_arr = gdal_asarray(dem_asc_file)
        
        # Create inundation mask (True where inundation depth > 0)
        inundation_mask = (potential_inun_arr > 0)
        inundated_cells_count = np.sum(inundation_mask)
        logger.info(f"Total inundated cells: {inundated_cells_count} out of {potential_inun_arr.size}")
        
        pixel_width = gdal_transform(max_inun_file)[1]
        num_cell_per_sample = int(sampling_dist / pixel_width)
        num_block_0 = (potential_inun_arr.shape[0] + num_cell_per_sample - 1) // num_cell_per_sample  # dir y (height)
        num_block_1 = (potential_inun_arr.shape[1] + num_cell_per_sample - 1) // num_cell_per_sample  # dir x (width)
    
        xOrigin, pw, xRot, yOrigin, ph, yRot  = gdal_transform(dem_asc_file) # affine transform parameters
        
        logger.info(f"Affine transform parameters DEM: {xOrigin, pw, xRot, yOrigin, ph, yRot}")
        logger.info(f"Pixel width: {pixel_width} m")
        logger.info(f"Sampling distance: {sampling_dist} m")
        logger.info(f"Number of cells in each block: {num_cell_per_sample}")
        logger.info(f"Potential inundation map shape: {potential_inun_arr.shape}")
        logger.info(f"DEM shape: {dem_arr.shape}")
        logger.info(f"Adjusted number of blocks in direction 0: {num_block_0}")
        logger.info(f"Adjusted number of blocks in direction 1: {num_block_1}")

        for i_0 in range(num_block_0):
            start_0 = i_0 * num_cell_per_sample
            
            # Ensure the block does not exceed the array bounds
            end_0 = min(start_0 + num_cell_per_sample, potential_inun_arr.shape[0])
            for i_1 in range(num_block_1):
                start_1 = i_1 * num_cell_per_sample
                
                # Ensure the block does not exceed the array bounds
                end_1 = min(start_1 + num_cell_per_sample, potential_inun_arr.shape[1])
                
                # Check if this block has any inundation
                inundation_block = inundation_mask[start_0:end_0, start_1:end_1]
                if not np.any(inundation_block):
                    logger.info(f"Block {i_0}, {i_1} has no inundation - skipping")
                    continue
                
                # Only process blocks that have inundation
                curr_dem_block_arr = dem_arr[start_0:end_0, start_1:end_1]
                
                # Create a masked version of the DEM where only inundated cells are considered
                # Use np.nan for non-inundated cells to exclude them from min calculation
                masked_dem = np.where(inundation_block, curr_dem_block_arr, np.nan)
                
                # Find the lowest point in the inundated area
                argmin_dem = np.nanargmin(masked_dem)
                logger.info(f"Minimum DEM value in inundated area of block {i_0}, {i_1}, argmin_dem: {argmin_dem} value: {masked_dem.ravel()[argmin_dem]}")
                
                # Calculate row and column indices correctly within the block
                block_height = end_0 - start_0
                block_width = end_1 - start_1
                
                # Fix: correctly convert flattened index to 2D coordinates
                # Row index = flattened index // width (not height)
                # Column index = flattened index % width
                ri = argmin_dem // block_width
                ci = argmin_dem % block_width
                
                logger.info(f"Row index: {ri}, Column index: {ci}")
                logger.info(f"row-> {start_0 + ri}, col-> {start_1 + ci}")
                
                # Additional bounds check to prevent index errors
                if start_0 + ri >= potential_inun_arr.shape[0] or start_1 + ci >= potential_inun_arr.shape[1]:
                    logger.warning(f"Calculated indices ({start_0 + ri}, {start_1 + ci}) out of bounds for array shape {potential_inun_arr.shape} - skipping")
                    continue
                    
                coords = rc2coords((xOrigin, pw, xRot, yOrigin, ph, yRot), (start_0 + ri, start_1 + ci))
                rl_coords_ls.append(coords)
                    
        logger.info(f"Found {len(rl_coords_ls)} representative locations in inundated areas")
        shp_output_file=f'ss_{sampling_dist}.shp'
        file_path = f'{self.work_dir}/{shp_output_file}'
        # Get bounds from the dem file and use that as the bounds for the shapefile
        save_pts_to_shp(rl_coords_ls, file_path)
        return file_path
    
def find_representative_locations_and_clusters(run_id, sampling_dist, n_clusters=10, random_state=42, n_init=10):
    work_dir = f"{OUTPUT_DIR}/rls"
    dem_asc_file = f"{CARLISLE_DATA_DIR}/Carlisle_5m.asc"
    simulation_dir = SIMULATION_DATA_DIR
    possible_inun_file = f"{simulation_dir}/Run1-0175.wd"
    run_meta_data_file = f"{work_dir}/run_meta_data.csv"
    
    rep_loc = RepLocation(work_dir)
    rl_file_path = rep_loc.spatial_sampling(run_id, dem_asc_file, possible_inun_file, sampling_dist=sampling_dist)
    
    # Create the cluster finder with GPU awareness
    cluster_finder = RLClusterFinder(work_dir, run_id, rl_file_path, sampling_dist, n_clusters=n_clusters,
                                    random_state=random_state, n_init=n_init)

    # Run clustering
    cluster_finder.run_clustering()
    
    # Save the meta data to csv file
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

