# Description: Data loader for U-Net model (PyTorch version)
import numpy as np
import os
import glob
import logging
import torch

from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import coords2rc, rc2coords, gdal_asarray, gdal_transform, gdal_writetiff
from modules.models.usrr_1dcnn.spatial_reduction_module.rep_location_finder import find_representative_locations_and_clusters
from modules.models.usrr_1dcnn.spatial_reduction_module.base_functions import read_shp_point
from modules.utils.run_util import check_device

logger = logging.getLogger("UNetDataManager")
logger.setLevel(logging.INFO)


class UNetDataManager():
    def __init__(self, sampling_dist, 
                 run_dir, 
                 epochs=10, 
                 batch_size=32, 
                 num_of_batch_per_map=6):

        self.rep_loc_file_path = f"{OUTPUT_DIR}/rls/ss_{sampling_dist}.shp"
        self.dem_asc_file = f"{CARLISLE_DATA_DIR}/Carlisle_5m.asc"
        self.simulation_dir = SIMULATION_DATA_DIR
        self.possible_inun_file = f"{self.simulation_dir}/Run1-0175.wd"
        self.reconstruction_out = f"{OUTPUT_DIR}/reconstruction.tif"
        self.area_check_file = f"{OUTPUT_DIR}/area_check.tif"
        
        inundation_files = sorted(glob.glob(f"{self.simulation_dir}/*.wd"))
        self.inundation_files = [f for f in inundation_files if self.get_timestep_index(f) > 7]
        
        # Parameters
        self.sampling_dist = sampling_dist
        self.run_dir = run_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.map_size = 64 # Should be divisible by 16 as the model is UNet
        self.cross_tile_dist_by_cell = 32 # For creating overlapping tiles
        self.num_of_batch_per_map = num_of_batch_per_map
        self.t_interval = 1
        self.dry_samples_per_map = 1
        self.device = check_device()
        
        self.test_event_ids = [1]
        self.val_event_ids = [9]
        self.train_event_ids = [2,3,4,5,6,7,8]
        self.prepare_input_template()
        self.prepare_indices()

    def get_representative_locations(self):
        if self.rep_loc_file_path and os.path.exists(self.rep_loc_file_path):
                # Load representative locations from file if provided
                rl_list = read_shp_point(self.rep_loc_file_path)
                logger.info(f"Loaded {len(rl_list)} representative locations from {self.rep_loc_file_path}")
        else:
            raise ValueError(f"RL file path is not valid: {self.rep_loc_file_path}")
        self.rl_list = rl_list
        return rl_list
    
    def reshaping_func_def(self, slicing_func, x, ax0_n, ax1_n):
        
        # First reshape the map to the desired size (a multiple of the map size)
        new_map_x = slicing_func(x)[:self.map_size * ax0_n, :self.map_size * ax1_n]
        new_map_x = new_map_x.reshape((-1, ax0_n, self.map_size)) 
        new_map_x = new_map_x.swapaxes(0, 1)
        
        new_map_x = new_map_x.reshape((ax0_n, -1, self.map_size, self.map_size))
        
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((-1, self.map_size, self.map_size))
        return new_map_x

    def build_tiling_func(self, cross_tile_size_r, cross_tile_size_c, development_mode=False):
        # Create a mask with maximum extent of the flood
        flood_extent = gdal_asarray(self.possible_inun_file)
        ext_mask = (~np.isnan(flood_extent)).astype(np.float32)
        
        # Create a slicing function to create tiles based on an offset
        slicing_func = lambda  x: x[cross_tile_size_r:, cross_tile_size_c:]
        new_shape = slicing_func(ext_mask).shape
        ax0_n = new_shape[0] // self.map_size  # new number of tiles in row
        ax1_n = new_shape[1] // self.map_size  # new number of tiles in column
        logger.info(f"New shape after slicing: {new_shape}, maps: {ax0_n}x{ax1_n}")
        
        # Create a reshaping function to get the maps(tiles)
        reshaping_func = lambda x: self.reshaping_func_def(slicing_func, x, ax0_n, ax1_n)
        
        # Create a mask to filter out the tiles that do not contain any flood or representative locations
        filter_arr = (np.sum(reshaping_func(ext_mask), axis=(1, 2)) !=0) \
            & (np.sum(reshaping_func(self.rl_map), axis=(1, 2)) != 0)
              
        if development_mode:
            return reshaping_func, filter_arr, ax0_n, ax1_n, (cross_tile_size_r, cross_tile_size_c)
        else: 
            tiling_func = lambda x: reshaping_func(x)[filter_arr, :, :]
            logger.info(f"Created tiling function successfully")
            return tiling_func
    
    def build_reshaping_func_seq(self, num_of_subm_maps_per_axes):
        # Function sequence to create reshaped maps(tiles) used for training
        self.reshaping_func_seq = []
        for r_i in range(num_of_subm_maps_per_axes):
            self.reshaping_func_seq.append(self.build_tiling_func(r_i * self.cross_tile_dist_by_cell,
                                                                  r_i * self.cross_tile_dist_by_cell))
    def tiles_prep_func_gpu(self, x):
        # Apply tiling functions to the input map
        tiles_maps =  [tiling_func(x) for tiling_func in self.reshaping_func_seq]
        
        logger.info(f"RL map: {x.shape}")
        logger.info(f"Number of tiles: {len(tiles_maps)}")
        
        # Concatenate the tiled maps along the first dimension
        tiles_maps = torch.cat(tiles_maps, dim=0)
        return tiles_maps
    
    def prepare_input_template(self):
        logger.info("Preparing input template for U-Net model")
        self.rl_list = self.get_representative_locations()
        transform  = gdal_transform(self.dem_asc_file)
        self.dem_map = gdal_asarray(self.dem_asc_file)
        
        # Convert RL coordinates to row/col indices
        self.rc_rl_points = [coords2rc(transform, rl) for rl in self.rl_list] 
        length_of_rl_points = len(self.rc_rl_points)
        logger.info(f"Number of representative locations: {length_of_rl_points}")
        
        # Create a binary map with representative locations. 1 at RL, 0 elsewhere
        self.rl_map = np.zeros(self.dem_map.shape)
        for row, col in self.rc_rl_points:
            self.rl_map[row, col] = 1
        
        # See the number of representative locations in the rl_map check for points with 1
        num_of_representative_locations = np.count_nonzero(self.rl_map)
        logger.info(f"Number of representative locations in map: {num_of_representative_locations}")

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
            self.shaped_dem = self.tiles_prep_func_gpu(self.dem_map)
            dem_mins = torch.amin(self.shaped_dem, dim=(1,2))
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
            
        # How many maps per event
        time_steps_per_event = 200 # Should be changed to dynamically read from the inundation files
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
    
    def get_batch(self, idx, test_mode=False, rl_depth=None):
        event_id = self.idx2eventid(idx)
        t_idx = self.idx2tidx(idx)
        batch_idx = self.idx2batchidx(idx)
        
        if test_mode:
            # Get the inundation map for the given event and timestep
            inundation_file = f'{self.simulation_dir}/Run{event_id}-{str(t_idx).zfill(4)}.wd'
            inundation_map = gdal_asarray(inundation_file)
            self.rl_map_filled = self.rl_map.copy()  # Make a copy to avoid modifying original
            rl_indices = np.where(self.rl_map_filled == 1)
            self.rl_map_filled[rl_indices[0], rl_indices[1]] = rl_depth
            self.rl_map_filled = torch.from_numpy(self.rl_map_filled).float().to(self.device)
            self.input_temp_filled = self.tiles_prep_func_gpu(self.rl_map_filled)
            input_filled = self.input_temp_filled[batch_idx * self.batch_size:
                                                         (batch_idx + 1) * self.batch_size, :, :].unsqueeze(1)
            dem_batch = torch.from_numpy(self.shaped_dem[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size,
                                             :, :].copy()).float().to(self.device).unsqueeze(1)
            return torch.cat((dem_batch, input_filled), dim=1), inundation_map
        with torch.no_grad():
            inundation_file = f'{self.simulation_dir}/Run{event_id}-{str(t_idx).zfill(4)}.wd'
            images = self.tiles_prep_func_gpu(self.process_inundation_file(inundation_file))
            images = images[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, :, :].unsqueeze(1)
            
            #Identify wet and dry tiles and prevent data imbalance
            wet_filter = torch.sum(images, dim=(1, 2, 3)) != 0
            dry_filter = torch.sum(~wet_filter)
            
            # If there is enough dry tiles, randomly select dry tiles
            if dry_filter.item() >=  self.dry_samples_per_map * 4:
                random_idxs = (np.random.uniform(0, 1, self.dry_samples_per_map) * dry_filter.item()).astype(int)
                wet_filter[torch.where(~wet_filter)[0]][random_idxs] = True
                
            # Get the corresponding DEM tile
            images = images[wet_filter]
            dem_images = torch.from_numpy(self.shaped_dem[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, :, :].copy())\
                .float().to(self.device).unsqueeze(1)
            dem_images = dem_images[wet_filter]
            
            input_filled =  torch.from_numpy(self.input_temp[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, :, :].copy())\
                .float().to(self.device).unsqueeze(1)
            input_filled = input_filled[wet_filter]
            
            # Fill the input with the DEM values
            input_filled[input_filled == 1] = images[input_filled == 1]
            return torch.cat((dem_images, input_filled), dim=1), images

    def get_timestep_index(self, filepath):
        filename = os.path.basename(filepath)
        parts = filename.split('-')
        if len(parts) < 2:
            return 0
        timestep_part = parts[1].split('.')[0]
        return int(timestep_part)
        
    def reconstruction_init(self):
        logger.info("Initializing reconstruction")
        with torch.no_grad():
            self.rl_map_filled = torch.from_numpy(self.rl_map.copy()).float().to(self.device)
        self.output_maps_list = []
        self.return_temp_list = []
        self.filter_list = []
        self.layers_seperation_idxs = []
        self.map_origins = []
        self.layer_sizes = []
        self.back_trans_funcs = []
        num_of_nodes_per_axes = round(self.map_size/self.cross_tile_dist_by_cell)
        start_idx = 0  
        with torch.no_grad():
            for r_i in range(num_of_nodes_per_axes):
                c_i = r_i
                reshaping_func, filter_arr, ax0_n, ax1_n, map_origin = \
                    self.build_tiling_func(r_i * self.cross_tile_dist_by_cell, 
                                           c_i * self.cross_tile_dist_by_cell, development_mode=True)
                # Number of unique tile with atleast one representative location
                self.layers_seperation_idxs.append((start_idx, start_idx + np.sum(filter_arr))) 
                start_idx = self.layers_seperation_idxs[-1][1]
                ouput_map = np.zeros(self.dem_map.size())
                self.output_maps_list.append(torch.from_numpy(ouput_map).float().to(self.device))
                self.return_temp_list.append(torch.from_numpy(reshaping_func(ouput_map)).float().to(self.device))
                self.filter_list.append(torch.from_numpy(filter_arr).bool().to(self.device))
                self.map_origins.append(map_origin)
                self.layer_sizes.append((ax0_n*self.map_size, ax1_n*self.map_size))
                self.back_trans_funcs.append(self.build_back_func_lambda(ax1_n))
                
            # Check if the reconstruction output covers the entire map
            temp_tensor = torch.ones(self.input_temp.shape).float().to(self.device)
            self.lyr_num_map = self.reconstruct_full_map_return_and_sum(temp_tensor)
            gdal_writetiff(self.lyr_num_map.detach().cpu().numpy(),self.area_check_file, ras_temp=self.dem_asc_file)
            
            ext_mask = gdal_asarray(self.possible_inun_file)
            ext_mask = ~np.isnan(ext_mask).astype(int)
            self.lyr_num_map[self.lyr_num_map == 0] = 1
            cond = np.sum((ext_mask==1) & (self.lyr_num_map.detach().cpu().numpy()==0))==0
            logger.info("Reconstruction output covers the entire map: " + str(cond))
        return 0

    def reconstruct_full_map_return_and_sum(self, x):
        for lyr_i in range(len(self.layers_seperation_idxs)):
            start_idx, end_idx = self.layers_seperation_idxs[lyr_i]
            #change the values of the output map in the return_temp_list
            self.return_temp_list[lyr_i][self.filter_list[lyr_i]] = x[start_idx:end_idx].squeeze(1) # remove axis with size 1
            row_size, col_size = self.layer_sizes[lyr_i]
            row_origin, col_origin = self.map_origins[lyr_i]
            self.output_maps_list[lyr_i][row_origin:row_origin + row_size, col_origin:col_origin + col_size] = \
                self.back_trans_funcs[lyr_i](self.return_temp_list[lyr_i])   
        output_sum = torch.sum(torch.stack(self.output_maps_list), dim=0)
        return output_sum
    
    def reconstruct_full_map(self, x):
        with torch.no_grad():
            output_final = torch.div(self.reconstruct_full_map_return_and_sum(x), self.lyr_num_map)
        
        return output_final

    def build_back_func_lambda(self, ax1_n):
        def back_func(x):
            x = x.reshape(-1,ax1_n, self.map_size, self.map_size)
            x = x.swapaxes(0, 1)
            x = x.reshape(ax1_n,-1, self.map_size)
            x = x.swapaxes(0, 1)
            x = x.reshape(-1, ax1_n * self.map_size)
            return x
        return lambda x: back_func(x)