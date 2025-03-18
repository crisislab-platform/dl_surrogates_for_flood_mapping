# Description: Data loader for U-Net model (PyTorch version)
import numpy as np
import os
import glob
import logging
import torch

from modules.utils.path_util import DATA_DIR
from torch.utils.data import Dataset
from modules.model_trainer.ussr_1dcnn.spatial_reduction_module.gdal_lib import coords2rc, rc2coords, gdal_asarray, gdal_transform
from modules.model_trainer.ussr_1dcnn.spatial_reduction_module.rep_location_finder import get_representative_locations
from modules.model_trainer.ussr_1dcnn.spatial_reduction_module.base_functions import read_shp_point
from modules.utils.run_util import check_device

logger = logging.getLogger("UNetDataManager")
logger.setLevel(logging.INFO)

class UNetDataManager():
    def __init__(self, sampling_dist, run_dir, epochs=10, batch_size=32, train_split=0.7, val_split=0.10, test_split=0.20, num_of_batch_per_map=4):
        self.sampling_dist = sampling_dist
        self.rep_loc_file_path = f"{DATA_DIR}/rls/ss_{sampling_dist}.shp"
        self.dem_asc_file = f"{DATA_DIR}/Carlisle_5m.asc"
        self.simulation_dir = f"{DATA_DIR}/DEM5m_2D/"
        self.possible_inun_file = f"{self.simulation_dir}/Run1-0175.wd"
        self.inundation_files = sorted(glob.glob(f"{self.simulation_dir}/*.wd"))
        self.pixel_size = None
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.run_dir = run_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.map_size = 64 # Should be divisible by 16
        self.number_of_classes = 1
        self.num_channels = 2  # DEM and point inundation
        self.reshaping_func_seq = None
        self.cross_tile_dist_by_cell = 31 # Increasing the sample size by overlapping the tiles
        self.rl_list = None
        self.rc_rl_points = None
        self.rc_rl_points_netCDF = None
        self.dem_map = None
        self.shaped_dem = None
        self.rl_map = None
        self.slice_row_padding = 0
        self.slice_col_padding = 0
        self.device = check_device()
        self.input_temp = None
        self.num_of_batch_per_map = num_of_batch_per_map
        self.t_interval = 1
        self.dry_samples_per_map = 1
        
        # Split events into train, validation and test
        self.test_event_ids = [1]
        self.val_event_ids = [9]
        self.train_event_ids = [i for i in range(1, 10) if i not in self.test_event_ids and i not in self.val_event_ids]
        
        self.prepare_input_template()
        self.prepare_indices()

    def get_representative_locations(self):
        if self.rep_loc_file_path and os.path.exists(self.rep_loc_file_path):
                # Load representative locations from file if provided
                rl_list = read_shp_point(self.rep_loc_file_path)
                logger.info(f"Loaded {len(rl_list)} representative locations from {self.rep_loc_file_path}")
        else:
            # Get representative locations using spatial sampling
            rl_list = get_representative_locations(self.run_dir, self.sampling_dist)
            logger.info(f"Generated {len(rl_list)} representative locations")
        self.rl_list = rl_list
        return rl_list
        
    def reshaping_func_def(self, slicing_func, x, ax0_n, ax1_n):
        logger.info(f"Reshaping function: {x.shape}, {ax0_n}, {ax1_n}")
        new_map_x = slicing_func(x)[:self.map_size * ax0_n, :self.map_size * ax1_n]
        new_map_x = new_map_x.reshape((-1, ax0_n, self.map_size))
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((ax0_n, -1, self.map_size, self.map_size))
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((-1, self.map_size, self.map_size))
        logger.info(f"Reshaped maps: {new_map_x.shape}")
        return new_map_x

    def build_tiling_func(self, cross_tile_size_r, cross_tile_size_c, development_mode=False):
        # Retrieve a tile from the original map
        
        # Create a mask with maximum extent of the flood
        flood_extent = gdal_asarray(self.possible_inun_file)
        ext_mask = (~np.isnan(flood_extent)).astype(np.float32)
        
        #focus on the flooded area
        slicing_func = lambda  x: x[cross_tile_size_r:, cross_tile_size_c:]
        
        # Create a slicing function to create tiles
        new_shape = slicing_func(ext_mask).shape
        ax0_n = new_shape[0] // self.map_size  # new number of maps in row
        ax1_n = new_shape[1] // self.map_size  # new number of maps in column
        logger.info(f"New shape after padding: {new_shape}, maps: {ax0_n}x{ax1_n}")
        
        # Create a reshaping function to get the maps
        reshaping_func = lambda x: self.reshaping_func_def(slicing_func, x, ax0_n, ax1_n)
        
        # Create a mask to filter out the tiles that do not contain any flood. 
        filter_arr = (np.sum(reshaping_func(ext_mask), axis=(1, 2)) !=0) \
            & (np.sum(reshaping_func(self.rl_map), axis=(1, 2)) != 0)
            
        # There is some probelm with the filter. It's filtering out most of the tiles. Check 
            
        if development_mode:
            return reshaping_func, filter_arr, ax0_n, ax1_n, cross_tile_size_r, cross_tile_size_c
        else: 
            tiling_func = lambda x: reshaping_func(x)[filter_arr, :, :]
            logger.info(f"Created tiling function successfully")
            return tiling_func
    
    def build_reshaping_func_seq(self, num_of_nodes_per_axes):
        self.reshaping_func_seq = []
        # Create tiles diagonally
        for r_i in range(num_of_nodes_per_axes):
            self.reshaping_func_seq.append(self.build_tiling_func(r_i * self.cross_tile_dist_by_cell, r_i * self.cross_tile_dist_by_cell))
        return 0
    
    def tiles_prep_func_gpu(self, x):
        # Apply tiling functions to the input map
        tiles_maps =  [tiling_func(x) for tiling_func in self.reshaping_func_seq]
        
        logger.info(f"RL map: {x.shape}")
        logger.info(f"Number of tiles: {len(tiles_maps)}")
        
        # Concatenate the tiled maps along the first dimension
        tiles_maps = torch.cat(tiles_maps, dim=0)
        logger
        return tiles_maps
    
    def prepare_input_template(self):
        logger.info("Preparing input template for U-Net model")
        
        self.rl_list = self.get_representative_locations()
        transform  = gdal_transform(self.dem_asc_file)
        self.dem_map = gdal_asarray(self.dem_asc_file)
        number_of_rows = self.dem_map.shape[0]
        
        # Convert RL coordinates to row/col indices
        self.rc_rl_points = [coords2rc(transform, rl) for rl in self.rl_list] 
        
        # Create a binary map with representative locations. 1 at RL, 0 elsewhere
        self.rl_map = np.zeros(self.dem_map.shape)
        self.rl_map[self.rc_rl_points[0], self.rc_rl_points[1]] = 1

        # Number of overlapping tiles per axes  y(row) and x(column)
        num_of_nodes_per_axes = round(self.map_size / self.cross_tile_dist_by_cell) 
        self.build_reshaping_func_seq(num_of_nodes_per_axes)
        logger.info("Preparation of reshaping functions completed")
        
        with torch.no_grad():
            # Set NaN values in DEM to max value if any
            is_dem_nan = np.isnan(self.dem_map).any()
            if is_dem_nan:
                logger.info("Setting NaN values in DEM to max value")
                self.dem_map[np.isnan(self.dem_map)] = np.nanmax(self.dem_map) + 1
                
            logger.info("Converting numpy arrays to tensors and moving to GPU")
            self.dem_map = torch.from_numpy(self.dem_map.copy()).float().to(self.device)
            self.rl_map =  torch.from_numpy(self.rl_map.copy()).float().to(self.device)
            
            self.input_temp = self.tiles_prep_func_gpu(self.rl_map)
            
            # Determine the batch size for training
            self.batch_size = self.input_temp.size()[0] // self.num_of_batch_per_map
            logger.info(f"input_temp shape: {self.input_temp.size()}")
            logger.info(f"batch_size: {self.batch_size}")
            
            assert self.input_temp.size()[0] % self.num_of_batch_per_map == 0,\
                f'Tiles ({self.input_temp.size()[0]}) of each map cannot be equally divided by the number of batches!' \
                f'Please adjust map_size or cross_tile_dist_by_cell.'
                
            # Elevation normalization for DEM
            # Create tiles
            self.shaped_dem = self.tiles_prep_func_gpu(self.dem_map)
            
            # Find the minimum value for each tile accross height and width
            dem_mins = torch.amin(self.shaped_dem, dim=(1,2))
            
            # Subtract the minimum value from each pixel in the tile
            self.shaped_dem = self.shaped_dem - dem_mins.repeat_interleave(self.map_size ** 2).view(self.shaped_dem.size())
            
            if is_dem_nan:
                # Set the max value to -1. These are typically pixels that were previously NaN
                self.shaped_dem[self.shaped_dem == self.shaped_dem.max()]  = -1
                self.dem_map[self.dem_map == self.dem_map.max()] = float('nan')
            
            # Move to CPU and convert tensors to numpy
            self.shaped_dem = self.shaped_dem.detach().to('cpu').numpy()
            self.input_temp = self.input_temp.detach().to('cpu').numpy()
            self.rl_map = self.rl_map.detach().to('cpu').numpy()
        torch.cuda.empty_cache()
        return 0
    
    def prepare_indices(self):
        """Creates indices for organising the training, testing and validation data accross different 
            events, timesteps, batches"""
            
        #How many maps per event
        time_steps_per_event = 280 # Should be changed
        self.maps_per_event = time_steps_per_event // self.t_interval
        
        # create indices for a given x and multiplier i
        idx_expander = lambda x, mul_i: np.repeat(x, mul_i) * mul_i + np.array(list(range(mul_i)) * len(x))
        
        test_map_idx = idx_expander(self.test_event_ids, self.maps_per_event)
        self.test_idxs = idx_expander(test_map_idx, self.num_of_batch_per_map)
        
        train_map_idx = idx_expander(self.train_event_ids, self.maps_per_event)
        self.train_idxs = idx_expander(train_map_idx, self.num_of_batch_per_map)

        rng = np.random.default_rng()
        rng.shuffle(self.train_idxs)
        rng.shuffle(self.test_idxs)
        val_map_index = idx_expander(self.val_event_ids, self.maps_per_event)
        self.val_idxs = idx_expander(val_map_index, self.num_of_batch_per_map)
        
        # Create lambda functions to get event_id, t_idx and batch_idx givem the idx
        self.idx2eventid = lambda idx: int((idx // self.num_of_batch_per_map) // self.maps_per_event)
        self.idx2tidx = lambda idx: int((idx // self.num_of_batch_per_map) % self.maps_per_event)
        self.idx2batchidx = lambda idx: int(idx % self.num_of_batch_per_map)  
        return 0
    
    def process_inundation_file(self, inundation_file):
        inundation_map = gdal_asarray(inundation_file)
        inundation_map = torch.from_numpy(inundation_map).float().to(self.device)
        return inundation_map
    
    def get_batch(self, idx, validation=False):
        event_id = self.idx2eventid(idx)
        t_idx = self.idx2tidx(idx)
        batch_idx = self.idx2batchidx(idx)
        
        if validation:
            return None
        
        with torch.no_grad():
            inundation_file = f'{self.simulation_dir}/Run{event_id}-{t_idx}.wd'
            images = self.tiles_prep_func_gpu(self.process_inundation_file(inundation_file))
            images = images[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, :, :].unsqueeze(1)
            
            #Identify wet and dry tiles and prevent data imbalance
            wet_filter = torch.sum(images, dim=(1, 2, 3)) != 0
            dry_filter = torch.sum(~wet_filter)
            
            # If there is enough dry tiles, randomly select dry tiles
            if dry_filter.item() >=  self.dry_samples_per_map * 4:
                random_idxs = (np.random.uniform(0, 1, self.dry_samples_per_map) * dry_filter.item()).astype(int)
                wet_filter[torch.where(~wet_filter)[0]][random_idxs] = True
                
            images = images[wet_filter]
            
            # Get the corresponding DEM tiles
            dem_images = torch.from_numpy(self.shaped_dem[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, :, :].copy())\
                .float().to(self.device).unsqueeze(1)
            dem_images = dem_images[wet_filter]
            
            input_filled =  torch.from_numpy(self.input_temp[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, :, :].copy())\
                .float().to(self.device).unsqueeze(1)
            input_filled = input_filled[wet_filter]
            
            # Add the actual inundation values to the input_filled
            input_filled[input_filled == 1] = images[input_filled == 1]
            
            # X, Y for batch
            return torch.cat((dem_images, input_filled), dim=1), images



