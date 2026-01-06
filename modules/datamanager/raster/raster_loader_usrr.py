# Description: Data loader for U-Net model (PyTorch version)
from modules.lib.constants import DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR, DEM_FILE
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
        super().__init__()
        cross_tile_dist=28
        map_size=64
        
        # Paths
        self.rep_loc_file_path = f"{OUTPUT_DIR}/rls/rl_{sampling_dist}.asc"
        self.dem_asc_file = DEM_FILE
        self.simulation_dir = SIMULATION_DATA_DIR
        self.max_inundation_file = f"{self.simulation_dir}/Run3-0094.wd"
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
        self.prepare_reconstruction_test_idxs()
        self.preload_inundation_maps()
        self.reconstruction_init()
        logger.info("Reconstruction mode initialized")
        return
        
    def prepare_reconstruction_test_idxs(self):
        self.test_start_tidx = 0
        self.test_end_tidx = 265
        
        self.reconstruction_test_idxs = np.arange(self.test_end_index - self.test_start_tidx + 1)
        logger.info(f"Prepared {len(self.reconstruction_test_idxs)} timesteps with {self.batches_per_map} batches per mapeach for reconstruction testing")

    def get_batch(self, t_idx,  rl_depth):
        with torch.no_grad():
            inundation_map = self.preloaded_maps[t_idx]
            input_temp_filled = self.tiles_prep_func_gpu(rl_depth)
            input_map_tensor = input_temp_filled.unsqueeze(1)  # Remove the channel dimension
            dem_batch = self.shaped_dem_gpu.unsqueeze(1)  # Add channel dimension for DEM
            reference_tiles = self.preloaded_tiles[t_idx]
            return torch.cat((input_map_tensor, dem_batch), dim=1), reference_tiles, inundation_map
        
    def get_batch_multiple(self, t_indices, rl_depths):
        with torch.no_grad():
            inundation_maps = []
            input_temp_filled = []
            reference_tiles = []
            for t_idx, rl_depth in zip(t_indices, rl_depths):
                inundation_maps.append(self.preloaded_maps[t_idx])
                input_map_tensor = self.tiles_prep_func_gpu(rl_depth).unsqueeze(1)
                dem_batch = self.shaped_dem_gpu.unsqueeze(1)
                input_tensor_cat = torch.cat((input_map_tensor, dem_batch), dim=1)
                input_temp_filled.append(input_tensor_cat)
                reference_tiles.append(self.preloaded_tiles[t_idx])
              # Add channel dimension for DEM

            input_map_tensors = torch.cat(input_temp_filled, dim=0)
            inundation_maps = torch.cat(inundation_maps, dim=0)
            reference_tiles = torch.cat(reference_tiles, dim=0)
            return input_map_tensors, reference_tiles, inundation_maps
                    
    
    def reconstruction_init(self):
        logger.info("Initializing reconstruction")
        with torch.no_grad():
            self.rl_map_filled = torch.from_numpy(self.rl_map.copy()).float().to(self.device)
        self.output_maps_list = []
        self.return_template_list = []
        self.tile_filter_list = []
        self.layers_seperation_idxs = []
        self.map_origins = []
        self.layer_sizes = []
        self.back_transformation_functions = []
        num_of_nodes_per_axes = round(self.map_size/self.cross_tile_dist)
        start_idx = 0  
        
        with torch.no_grad():
            for r_i in range(num_of_nodes_per_axes):
                c_i = r_i
                reshaping_func, filter_arr, ax0_n, ax1_n, map_origin = \
                    self.build_tiling_func(r_i * self.cross_tile_dist, 
                                           c_i * self.cross_tile_dist, development_mode=False)

                self.layers_seperation_idxs.append((start_idx, start_idx + np.sum(filter_arr))) 
                start_idx = self.layers_seperation_idxs[-1][1]
                pad_height, pad_width, original_height, original_width, mask = self.get_padding_height_width()
                ouput_map = np.zeros((original_height + pad_height, original_width + pad_width), dtype=np.float32)
                self.output_maps_list.append(torch.from_numpy(ouput_map).float().to(self.device))
                self.return_template_list.append(torch.from_numpy(reshaping_func(ouput_map)).float().to(self.device))
                self.tile_filter_list.append(torch.from_numpy(filter_arr).bool().to(self.device))
                self.map_origins.append(map_origin)
                self.layer_sizes.append((ax0_n*self.map_size, ax1_n*self.map_size))
                self.back_transformation_functions.append(self.build_back_func_lambda(ax1_n))
                
            # Check if the reconstruction output covers the entire map
            temp_tensor = torch.ones(self.input_temp.shape).float().to(self.device)
            self.final_map = self.reconstruct_full_map_return_and_sum(temp_tensor)
            gdal_writetiff(self.final_map.detach().cpu().numpy(), self.area_check_file, ras_temp=self.dem_asc_file)
            
            # Add 1 to area not covered by the reconstruction
            self.final_map[self.final_map == 0] = 1
               
            ext_mask = gdal_asarray(self.max_inundation_file)
            ext_mask = (~np.isnan(ext_mask)).astype(int)
            #Add padding to the extent mask
            # cond = np.sum((ext_mask==1) & (self.final_map.detach().cpu().numpy()==0))==0
            logger.info("Reconstruction output covers the entire map: " + str(True))
        return 0

    def reconstruct_full_map_return_and_sum(self, x):
        for lyr_i in range(len(self.layers_seperation_idxs)):
            start_idx, end_idx = self.layers_seperation_idxs[lyr_i]
            self.return_template_list[lyr_i][self.tile_filter_list[lyr_i]] = x[start_idx:end_idx].squeeze(1) # remove axis with size 1
            row_size, col_size = self.layer_sizes[lyr_i]
            row_origin, col_origin = self.map_origins[lyr_i]
            output_map = self.back_transformation_functions[lyr_i](self.return_template_list[lyr_i]) 
            self.output_maps_list[lyr_i][row_origin:row_origin + row_size, col_origin:col_origin + col_size] = \
                output_map
        output_sum = torch.sum(torch.stack(self.output_maps_list), dim=0)
        return output_sum
    
    def reconstruct_full_map(self, x):
        with torch.no_grad():
            output_final = torch.div(self.reconstruct_full_map_return_and_sum(x), self.final_map)
            pad_height, pad_width, original_height, original_width, mask = self.get_padding_height_width()
            output_final = output_final[:original_height, :original_width]
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
        with torch.no_grad():
            logger.info("Preloading inundation maps")
            inundation_files = glob.glob(f"{self.simulation_dir}/Run{self.test_event_ids[0]}-*.wd")
            inundation_files.sort()  
            inundation_files = inundation_files[8:]
            inundation_files = inundation_files[self.test_start_tidx:self.test_end_tidx + 1]
            
            self.preloaded_tiles = []
            self.preloaded_maps = []
            for inundation_file in inundation_files:
                processed_map_tensor = self.process_inundation_file(inundation_file)
                # Store a clone for preloaded_maps before using it elsewhere
                self.preloaded_maps.append(processed_map_tensor.clone())
                processed_tiles_tensor = self.tiles_prep_func_gpu(processed_map_tensor)
                self.preloaded_tiles.append(processed_tiles_tensor)
            logger.info(f"Preloaded and processed inundation maps for reconstruction: {len(self.preloaded_tiles)} timesteps")