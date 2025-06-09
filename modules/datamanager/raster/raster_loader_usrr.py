# Description: Data loader for U-Net model (PyTorch version)
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import coords2rc, rc2coords, gdal_asarray, gdal_transform, gdal_writetiff
from modules.datamanager.raster.raster_loader_unet import UNetDataManager
from modules.models.usrr_1dcnn.lib.base_functions import read_shp_point
from modules.utils.run_util import check_device
from modules.datamanager.raster.raster_loader_base import USRRDataManager

import numpy as np
import os
import glob
import logging
import torch
import pandas as pd
from modules.datamanager.datamanager import DataManager

logger = logging.getLogger("UNetDataManager")
logger.setLevel(logging.INFO)

class ReconsturctionDataManager(USRRDataManager):
    
    def __init__(self, sampling_dist, 
                 run_dir, 
                 batches_per_map=6):
        super().__init__(run_dir=run_dir, sampling_dist=sampling_dist, batches_per_map=batches_per_map)
        cross_tile_dist=32
        map_size=64
        
        # Paths
        self.rep_loc_file_path = f"{OUTPUT_DIR}/rls/ss_{sampling_dist}.csv"
        self.dem_asc_file = f"{SIMULATION_DATA_DIR}/Carlisle_5m.asc"
        self.simulation_dir = SIMULATION_DATA_DIR
        self.possible_inun_file = f"{self.simulation_dir}/Run1-0145.wd"
        self.area_check_file = f"{OUTPUT_DIR}/area_check.tif"
        
        # Hyperparameters and configurations
        self.sampling_dist = sampling_dist
        self.run_dir = run_dir
        self.batch_size = None
        self.map_size = map_size # Should be divisible by 16 as the model is UNet
        self.cross_tile_dist = cross_tile_dist # For creating overlapping tiles. The distance between the tiles in cells
        self.batches_per_map = batches_per_map # Number of batches for one map (timestep)
        self.t_interval = 1
        self.device = check_device()
        self.test_event_ids = [1]
        
        self.prepare_input_template()
        self.preload_inundation_maps()
        self.prepare_reconstruction_test_idxs()
        self.reconstruction_init()
        logger.info("Reconstruction mode initialized")
        return
        
    def prepare_reconstruction_test_idxs(self):
        test_start_tidx = (17 * 4) - (2 * 4) - 1 
        test_end_tidx = (65 * 4) - (2 * 4) - 1
        
        reconstruction_test_idxs = []
        for t_idx in range(test_start_tidx, test_end_tidx + 1):
            tiles_per_batch = len(self.preloaded_tiles[t_idx]) // self.batches_per_map
            timestep_batches = []
            # Create batches_per_map number of batches for this timestep
            for batch_idx in range(self.batches_per_map):
                start_idx = batch_idx * tiles_per_batch
                end_idx = (batch_idx + 1) * tiles_per_batch
                
                # Create batch indices for this specific batch in the timestep
                batch_indices = np.arange(start_idx, end_idx)
                timestep_batches.append(batch_indices)
                
            # Store all batches for this timestep
            self.reconstruction_test_idxs.append(timestep_batches)
        
        self.reconsytruction_test_idxs = np.array(reconstruction_test_idxs)
        self.reconstruction_test_idxs = self.reconstruction_test_idxs.reshape(-1, self.batches_per_map)
        logger.info(f"Prepared {len(self.reconstruction_test_idxs)} timesteps with {self.batches_per_map} batches each for reconstruction testing")

    def get_batch(self, t_idx, batch_indices, rl_depth):
        inundation_map = self.preloaded_maps[t_idx]
        rl_map_filled = torch.from_numpy(rl_depth).float().to(self.device)
        input_temp_filled = self.tiles_prep_func_gpu(rl_map_filled)
        input_map_tensor = input_temp_filled[batch_indices].unsqueeze(1)
        dem_batch = self.shaped_dem_gpu[batch_indices].unsqueeze(1)
        return torch.cat((input_map_tensor, dem_batch), dim=1), inundation_map
                 

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
        # assign 0 to values less than 0.3
        inundation_map[inundation_map < 0.3] = 0
        inundation_map = torch.from_numpy(inundation_map).float().to(self.device)
        return inundation_map
    
        
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
        num_of_nodes_per_axes = round(self.map_size/self.cross_tile_dist)
        start_idx = 0  
        with torch.no_grad():
            for r_i in range(num_of_nodes_per_axes):
                c_i = r_i
                reshaping_func, filter_arr, ax0_n, ax1_n, map_origin = \
                    self.build_tiling_func(r_i * self.cross_tile_dist, 
                                           c_i * self.cross_tile_dist, development_mode=True)
                #Number of unique tiles with at least one representative location
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
            gdal_writetiff(self.lyr_num_map.detach().cpu().numpy(), self.area_check_file, ras_temp=self.dem_asc_file)
            
            # Add 1 to area not covered by the reconstruction
            self.lyr_num_map[self.lyr_num_map == 0] = 1
               
            ext_mask = gdal_asarray(self.possible_inun_file)
            ext_mask = (~np.isnan(ext_mask)).astype(int)
         
            cond = np.sum((ext_mask==1) & (self.lyr_num_map.detach().cpu().numpy()==0))==0
            logger.info("Reconstruction output covers the entire map: " + str(cond))
        return 0

    def reconstruct_full_map_return_and_sum(self, x):
        for lyr_i in range(len(self.layers_seperation_idxs)-1):
            start_idx, end_idx = self.layers_seperation_idxs[lyr_i]
            self.return_temp_list[lyr_i][self.filter_list[lyr_i]] = x[start_idx:end_idx].squeeze(1) # remove axis with size 1
            row_size, col_size = self.layer_sizes[lyr_i]
            row_origin, col_origin = self.map_origins[lyr_i]
            self.output_maps_list[lyr_i][row_origin:row_origin + row_size, col_origin:col_origin + col_size] = \
                self.back_trans_funcs[lyr_i](self.return_temp_list[lyr_i])   
        # output_sum = torch.sum(torch.stack(self.output_maps_list), dim=0)
        output_sum = self.output_maps_list[0]
        return output_sum
    
    def reconstruct_full_map(self, x):
        with torch.no_grad():
            # output_final = torch.div(self.reconstruct_full_map_return_and_sum(x), self.lyr_num_map)
            output_final = self.reconstruct_full_map_return_and_sum(x)
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
    
    def preload_inundation_maps(self):
        logger.info("Preloading inundation maps")
        inundation_files = glob.glob(f"{self.simulation_dir}/Run{self.test_event_ids[0]}-*.wd")
        inundation_files.sort()  
        inundation_files = inundation_files[8:]
        
        self.preloaded_tiles = []
        self.preloaded_maps = []
        for inundation_file in inundation_files:
            processed_map_tensor = self.process_inundation_file(inundation_file)
            self.preloaded_maps.append(processed_map_tensor)
            processed_tiles_tensor = self.tiles_prep_func_gpu(processed_map_tensor)
            self.preloaded_tiles.append(processed_tiles_tensor)
        logger.info(f"Preloaded and processed inundation maps for reconstruction: {len(self.preloaded_tiles)} timesteps")