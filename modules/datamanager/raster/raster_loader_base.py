from modules.datamanager.datamanager import DataManager
from modules.models.usrr_1dcnn.lib.gdal_lib import coords2rc,gdal_asarray, gdal_transform
import torch
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USRRDataManager")


class USRRDataManager(DataManager):
    def __init__(self):
        super().__init__()

    def build_reshaping_func_seq(self, num_of_subm_maps_per_axes):
        # Function sequence to create reshaped maps(tiles) used for training
        self.reshaping_func_seq = []
        for r_i in range(num_of_subm_maps_per_axes):
            self.reshaping_func_seq.append(self.build_tiling_func(r_i * self.cross_tile_dist,
                                                                  r_i * self.cross_tile_dist))
    def prepare_input_template(self):
        logger.info("Preparing input template for U-Net model")
        rl_map_np, dem_map_np = self.get_representative_locations()
        
        # See the number of representative locations in the rl_map check for points with 1
        num_of_representative_locations = np.count_nonzero(rl_map_np)
        logger.info(f"Number of representative locations in map: {num_of_representative_locations}")

        # Number of overlapping tiles per axes  y(row) and x(column)
        num_of_nodes_per_axes = round(self.map_size / self.cross_tile_dist) 
        
        # Keep a CPU copy of rl_map for tiling function building
        self.rl_map = rl_map_np
        self.build_reshaping_func_seq(num_of_nodes_per_axes)
        logger.info("Preparation of reshaping functions completed")
        
        with torch.no_grad():
            # Handle NaN values in DEM
            is_dem_nan = np.isnan(dem_map_np).any()
            if is_dem_nan:
                logger.info("Setting NaN values in DEM to max value")
                dem_map_np[np.isnan(dem_map_np)] = np.nanmax(dem_map_np) + 1
                
            logger.info("Converting numpy arrays to tensors and moving to GPU")
            # Move data to GPU and keep it there
            self.dem_map = torch.from_numpy(dem_map_np.copy()).float().to(self.device)
            self.rl_map_gpu = torch.from_numpy(rl_map_np.copy()).float().to(self.device)
            self.input_temp = self.tiles_prep_func_gpu(self.rl_map_gpu)
            
            # Determine the batch size for training
            self.batch_size = self.input_temp.size()[0] // self.batches_per_map
            logger.info(f"input_temp shape: {self.input_temp.size()}")
            logger.info(f"batch_size: {self.batch_size}")
            
            assert self.input_temp.size()[0] % self.batches_per_map == 0,\
                f'Tiles ({self.input_temp.size()[0]}) of each map cannot be equally divided by the number of batches!' \
                f'Please adjust map_size or cross_tile_dist_by_cell.'
                
            # Elevation normalization for DEM - keep on GPU
            self.shaped_dem = self.tiles_prep_func_gpu(self.dem_map)
            dem_mins = torch.amin(self.shaped_dem, dim=(1,2))
            self.shaped_dem = self.shaped_dem - dem_mins.repeat_interleave(self.map_size ** 2).view(self.shaped_dem.size())
            
            if is_dem_nan:
                # Set the max value to -1. These are typically pixels that were previously NaN
                self.shaped_dem[self.shaped_dem == self.shaped_dem.max()] = -1
                self.dem_map[self.dem_map == self.dem_map.max()] = float('nan')
            
            # Store GPU copies for fast access during training
            self.shaped_dem_gpu = self.shaped_dem
            self.input_temp_gpu = self.input_temp
            
            # Also keep CPU copies for compatibility with existing code
            self.shaped_dem_cpu = self.shaped_dem.detach().cpu().numpy()
            self.input_temp_cpu = self.input_temp.detach().cpu().numpy()
            self.rl_map_cpu = self.rl_map_gpu.detach().cpu().numpy()
            
            # Update references to use CPU copies where needed
            self.shaped_dem = self.shaped_dem_cpu
            self.input_temp = self.input_temp_cpu
            self.rl_map = self.rl_map_cpu
            
        logger.info("Input template preparation complete with optimized GPU usage")
        torch.cuda.empty_cache()
        return 0
    
    def process_inundation_file(self, inundation_file):
        inundation_map = gdal_asarray(inundation_file)
        #check if there are any NaN values in the inundation map
        if np.isnan(inundation_map).any():
            logger.info(f"Setting NaN values in inundation map to 0")
            inundation_map[np.isnan(inundation_map)] = 0
        
        #check if there are any negative values in the inundation map
        if np.any(inundation_map < 0):
            logger.info(f"Setting negative values in inundation map to 0")
            inundation_map[inundation_map < 0] = 0
        inundation_map = torch.from_numpy(inundation_map).float().to(self.device)
        return inundation_map
    
    def get_representative_locations(self):
        rl_list = pd.read_csv(self.rep_loc_file_path)
        logger.info(f"rep loc: {rl_list.iloc[0]}")
        rl_list = rl_list[['x', 'y']].values
        rl_list = [tuple(x) for x in rl_list]
        
        transform = gdal_transform(self.dem_asc_file)
        dem_map_np = gdal_asarray(self.dem_asc_file)

        # Convert RL coordinates to row/col indices
        self.rc_rl_points = [coords2rc(transform, rl) for rl in rl_list] 
        transform = gdal_transform(self.dem_asc_file)
        dem_map_np = gdal_asarray(self.dem_asc_file)
        
        # Convert RL coordinates to row/col indices
        rc_rl_points = [coords2rc(transform, rl) for rl in  rl_list] 
        rl_map_np = np.zeros(dem_map_np.shape)
        for row, col in rc_rl_points:
            rl_map_np[row, col] = 1
        
        logger.info(f"Loaded {len(rl_list)} representative locations ")
        return rl_map_np, dem_map_np
    
    def tiles_prep_func_gpu(self, x):
        tiles_maps =  [tiling_func(x) for tiling_func in self.reshaping_func_seq]
        tiles_maps = torch.cat(tiles_maps, dim=0)
        return tiles_maps
    
    def build_tiling_func(self, cross_tile_size_r, cross_tile_size_c, development_mode=False):
        
        # Create a mask with maximum extent of the flood
        flood_extent = gdal_asarray(self.possible_inun_file)
        ext_mask = (~np.isnan(flood_extent)).astype(np.float32)
        
        # Create a slicing function to create tiles based on an offset
        slicing_func = lambda  x: x[3 + cross_tile_size_r:, 23 + cross_tile_size_c:]
        new_shape = slicing_func(ext_mask).shape
        ax0_n = new_shape[0] // self.map_size  # new number of tiles in row
        ax1_n = new_shape[1] // self.map_size  # new number of tiles in column
        logger.info(f"New shape after slicing: {new_shape}, maps: {ax0_n}x{ax1_n}")
        
        # Create a reshaping function to get the maps(tiles)         
        # Create a mask to filter out the tiles that do not contain any flood or representative locations
        reshaping_func = lambda x: self.reshaping_func_def(slicing_func, x, ax0_n, ax1_n)
        filter_arr = (np.sum(reshaping_func(ext_mask), axis=(1, 2)) !=0) \
            & (np.sum(reshaping_func(self.rl_map), axis=(1, 2)) != 0)
              
        if development_mode:
            return reshaping_func, filter_arr, ax0_n, ax1_n, (3 + cross_tile_size_r, 23 + cross_tile_size_c)
        else: 
            tiling_func = lambda x: reshaping_func(x)[filter_arr, :, :]
            logger.info(f"Created tiling function successfully")
            return tiling_func
    
    def reshaping_func_def(self, slicing_func, x, ax0_n, ax1_n):
        # First reshape the map to the desired size (a multiple of the map size)
        new_map_x = slicing_func(x)[:self.map_size * ax0_n, :self.map_size * ax1_n]
        new_map_x = new_map_x.reshape((-1, ax1_n, self.map_size)) 
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((ax1_n, -1, self.map_size, self.map_size))
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((-1, self.map_size, self.map_size))
        return new_map_x


    