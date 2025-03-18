from modules.utils.path_util import DATA_DIR
from pathlib import Path
from modules.model_trainer.ussr_1dcnn.spatial_reduction_module.base_functions import *
import logging
import numpy as np
from modules.model_trainer.ussr_1dcnn.spatial_reduction_module.gdal_lib import gdal_asarray, gdal_transform, rc2coords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Representaitve_Location_Finder")

class RepLocation:
    def __init__(self, results_dir):
        self.work_dir = results_dir
        logger.info(f"Representative Location Finder initialized with work directory: {self.work_dir}")
        
    def spatial_sampling(self, dem_asc_file, max_inun_file, sampling_dist,
                         save2shp=False, shp_output_file=None):
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
                ri = argmin_dem // block_height
                ci = argmin_dem % block_width
                
                logger.info(f"Row index: {ri}, Column index: {ci}")
                logger.info(f"row-> {start_0 + ri}, col-> {start_1 + ci}")
                coords = rc2coords((xOrigin, pw, xRot, yOrigin, ph, yRot), (start_0 + ri, start_1 + ci))
                rl_coords_ls.append(coords)
                    
        logger.info(f"Found {len(rl_coords_ls)} representative locations in inundated areas")
        if save2shp:
            assert shp_output_file is not None, 'Please specify output shapefile name!'
            # Get bounds from the dem file and use that as the bounds for the shapefile
            save_pts_to_shp(rl_coords_ls, f'{self.work_dir}/{shp_output_file}')
            
        return rl_coords_ls

def test_rl_selection(run_dir, sampling_dist):
    return get_representative_locations(run_dir, sampling_dist)

def get_representative_locations(run_dir, sampling_dist):
    work_dir = run_dir
    # test inundation extent reconstruction accuracy based on 2D linear interpolation
    dem_asc_file = f"{DATA_DIR}/Carlisle_5m.asc"
    simulation_dir = f"{DATA_DIR}/DEM5m_2D/"
    possible_inun_file = f"{simulation_dir}/Run1-0175.wd"
    
    rep_loc = RepLocation(work_dir)
    rl_list = rep_loc.spatial_sampling(dem_asc_file, possible_inun_file, sampling_dist=sampling_dist,
                                      save2shp=True, shp_output_file=f'ss_{sampling_dist}.shp')
    
    return rl_list


