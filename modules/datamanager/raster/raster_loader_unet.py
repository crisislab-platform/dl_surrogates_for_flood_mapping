# Description: Data loader for U-Net model (PyTorch version)
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import coords2rc, rc2coords, gdal_asarray, gdal_transform, gdal_writetiff
from modules.models.usrr_1dcnn.lib.base_functions import read_shp_point
from modules.utils.run_util import check_device

import numpy as np
import os
import glob
import logging
import torch
import pandas as pd
from modules.datamanager.raster.raster_loader_base import USRRDataManager

logger = logging.getLogger("UNetDataManager")
logger.setLevel(logging.INFO)


class UNetDataManager(USRRDataManager):
    
    def __init__(self, sampling_dist, 
                 run_dir, 
                 batches_per_map=6):
        
        super().__init__()
        
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
        self.dry_samples_per_map = 1
        self.device = check_device()
        self.test_event_ids = [1]
        self.train_event_ids = [2,3,4,5,6,7,8,9]
        self.all_event_ids = self.test_event_ids + self.train_event_ids
        
        self.prepare_input_template()
        self.preload_inundation_maps()
        self.prepare_batch_idxs()
        self.prepare_test_event_batches()
        
    def prepare_test_event_batches(self):
        logger.info("Preparing test event data for UNet evaluation")
        
        # Calculate time indices for the test period
        test_start_tidx = (17 * 4) - (2 * 4) - 1 
        test_end_tidx = (65 * 4) - (2 * 4) - 1
        test_event_id = self.test_event_ids[0]
        
        # Create containers for final batch structures
        self.test_input_batches = []
        self.test_output_batches = []
        
        # Process each timestep in the test range and create batches directly
        for t_idx in range(test_start_tidx, test_end_tidx + 1):
            if t_idx not in self.preloaded_tiles[test_event_id]:
                logger.warning(f"Timestep {t_idx} not found in preloaded tiles for event {test_event_id}")
                continue
                
            # Get preloaded tiles for this timestep
            output_tiles = self.preloaded_tiles[test_event_id][t_idx]
            
            # Create sparse RL input tensor
            rl_tensor = torch.zeros_like(output_tiles, device=self.device)
            rl_tensor[self.input_temp_gpu.bool()] = output_tiles[self.input_temp_gpu.bool()]
            
            # Create and store batches directly
            timestep_batches = []
            timestep_output_batches = []
            
            # Use batches_per_map directly instead of calculating from size
            tiles_per_batch = output_tiles.size(0) // self.batches_per_map
            
            # Create exactly batches_per_map batches for this timestep
            for b_idx in range(self.batches_per_map):
                start_idx = b_idx * tiles_per_batch
                end_idx = (b_idx + 1) * tiles_per_batch
                
                # Create input tensor (RL values + DEM)
                input_batch = rl_tensor[start_idx:end_idx].unsqueeze(1)
                dem_batch = self.shaped_dem_gpu[start_idx:end_idx].unsqueeze(1)
                model_input = torch.cat((input_batch, dem_batch), dim=1)
                
                # Get corresponding output batch
                output_batch = output_tiles[start_idx:end_idx].unsqueeze(1)
                
                # Store batches
                timestep_batches.append(model_input)
                timestep_output_batches.append(output_batch)
                
            # Add batches for this timestep to main containers
            if timestep_batches:
                self.test_input_batches.append(timestep_batches)
                self.test_output_batches.append(timestep_output_batches)
        
        logger.info(f"Prepared batch structure for {len(self.test_input_batches)} timesteps with matching inputs and outputs")
    
        
    def get_maps_per_event(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_dir}/Run{event_id}-*.wd")
        inundation_files.sort()
        inundation_files = inundation_files[8:]  # Skip the first 8 timesteps
        return len(inundation_files)
    
    def idx_expander(self, event_ids):
        all_indices = []
        if len(event_ids) == 0:
            return np.array([])
        
        # First, collect all valid indices from each event
        for event_id in event_ids:
            start_idx = self.event_start_id_map[event_id]
            num_idx = self.get_maps_per_event(event_id) * self.batches_per_map
            event_indices = start_idx + np.arange(num_idx)
            all_indices.extend(event_indices)
            
        # Convert to numpy array and shuffle to mix events
        all_indices = np.array(all_indices)
        rng = np.random.default_rng(341)  # Use same seed for consistency
        rng.shuffle(all_indices)
        
        # Create batches from mixed indices
        num_of_batches = len(all_indices) // self.batch_size
        idx_batch_list = []
        
        for i in range(num_of_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(all_indices))
            if batch_end - batch_start < self.batch_size:
                # Skip incomplete batches
                continue
            batch_indices = all_indices[batch_start:batch_end]
            idx_batch_list.append(batch_indices)
                
        if len(idx_batch_list) > 0:
            idx_batch_list = np.array(idx_batch_list)
            logger.info(f"Index batches shape: {idx_batch_list.shape}")
            logger.info(f"Created mixed-event batches with data from multiple events")
            return idx_batch_list
        else:
            logger.error("No batches were created! Check your data and batch size.")
            return np.array([])
        
    
    def prepare_batch_idxs(self):
        #start index for each event
        self.event_start_id_map = {}
        
        # Calculate start indices for each event
        for event_id in self.all_event_ids:
            if event_id == 1:  # First event
                start_idx = 0
            else:
                prev_event_id = event_id - 1
                start_idx = self.event_start_id_map[prev_event_id] + self.get_maps_per_event(prev_event_id) * self.batches_per_map
            self.event_start_id_map[event_id] = start_idx
        
        self.train_idx = self.idx_expander(self.train_event_ids)
        # self.validation_idx = self.idx_expander(self.val_event_ids)
        self.test_idx = self.idx_expander(self.test_event_ids)      
    
        return 0
    
    def find_local_indices(self, event_id, idxs):
        tile_indices = []
        for idx in idxs:
            if event_id in self.event_start_id_map:
                start_idx = self.event_start_id_map[event_id]
                # Calculate local index within this event
                local_idx = idx - start_idx
                # Calculate which map/timestep this index belongs to
                map_idx = local_idx // self.batches_per_map
                # Calculate which batch within the map this index belongs to
                batch_idx = local_idx % self.batches_per_map
                
                # Make sure the indices are valid for this event
                maps_in_event = self.get_maps_per_event(event_id)
                if 0 <= map_idx < maps_in_event:
                    tile_indices.append((map_idx, batch_idx))
        return tile_indices

    def find_event_id(self, idxs):
        event_indices = {}
        for idx in idxs:
            for event_id in self.all_event_ids:
                start_idx = self.event_start_id_map[event_id]
                end_idx = start_idx + self.get_maps_per_event(event_id) * self.batches_per_map - 1
                if start_idx <= idx <= end_idx:
                    if event_id not in event_indices:
                        event_indices[event_id] = []
                    event_indices[event_id].append(idx)
                    break
        return event_indices
        
    def get_batch(self, idxs):
        event_indices_map = self.find_event_id(idxs)
        all_inputs = []
        all_outputs = []
        for event_id, idx_list in event_indices_map.items():
            local_indices = self.find_local_indices(event_id, idx_list)
            for t_idx, batch_idx in local_indices:
                #Use preloaded tiles instead of loading from file
                if t_idx in self.preloaded_tiles[event_id]:
                    images = self.preloaded_tiles[event_id][t_idx]
                    images = images[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size, :, :].unsqueeze(1)
                else:
                    logger.warning(f"Preloaded tiles not available for event {event_id}, loading from file")
                    exit(1)
                
                # Identify wet and dry tiles and prevent data imbalance
                wet_filter = torch.sum(images, dim=(1, 2, 3)) != 0
                dry_filter = torch.sum(~wet_filter)
                
                # If there are enough dry tiles, randomly select some dry tiles
                if dry_filter.item() >= self.dry_samples_per_map * 4:
                    random_idxs = (np.random.uniform(0, 1, self.dry_samples_per_map) * dry_filter.item()).astype(int)
                    wet_filter[torch.where(~wet_filter)[0][random_idxs]] = True
                
                # Filter the relevant tiles
                # Get the corresponding DEM tiles - use GPU version if available
                filtered_images = images[wet_filter]
                dem_images = self.shaped_dem_gpu[batch_idx * self.batch_size: 
                                                  (batch_idx + 1) * self.batch_size, :, :].unsqueeze(1)

                dem_images = dem_images[wet_filter]
                input_filled = self.input_temp_gpu[batch_idx * self.batch_size: 
                                                    (batch_idx + 1) * self.batch_size, :, :].unsqueeze(1)

                input_filled = input_filled[wet_filter]
                input_filled[input_filled == 1] = filtered_images[input_filled == 1]
                
                #Add to batch collections
                all_inputs.append(torch.cat((input_filled, dem_images), dim=1))
                all_outputs.append(filtered_images)
        
        #Concatenate all batches from different events
        if all_inputs and all_outputs:
            return torch.cat(all_inputs, dim=0), torch.cat(all_outputs, dim=0)
        else:
            logger.error("No valid batches collected")
            return None, None
                
    def preload_inundation_maps(self):
        logger.info("Preloading inundation maps")
        inundation_files = glob.glob(f"{self.simulation_dir}/Run*-*.wd")
        inundation_files.sort()
        inundation_files = inundation_files[8:]
        
        # Preload inundation maps for all events
        self.preloaded_tiles = {}
        for inundation_file in inundation_files:
            # Extract event_id and timestep
            filename = os.path.basename(inundation_file)
            parts = filename.split('-')
            event_id = int(parts[0].replace("Run", ""))
            t_idx = int(parts[1].split('.')[0])
            
            # Process inundation map and convert to tiles
            processed_map = self.process_inundation_file(inundation_file)
            processed_tiles = self.tiles_prep_func_gpu(processed_map)
            
            # Store processed tiles by event_id and timestep
            if event_id not in self.preloaded_tiles:
                self.preloaded_tiles[event_id] = {}
            self.preloaded_tiles[event_id][t_idx] = processed_tiles
            
        logger.info(f"Preloaded and processed inundation maps for {len(self.preloaded_tiles)} events")
        return 0