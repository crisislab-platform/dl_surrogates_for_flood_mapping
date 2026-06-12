import glob
import logging
import os
import numpy as np
import rasterio as rio
import torch
from modules.lib.constants import DEM_FILE, STUDY_AREA, BC_DATA_DIR, FLOOD_MAPS_DIR
from modules.datamanager.datamanager import DataManager
from modules.utils.run_util import check_device
from modules.lib.gdal_lib import gdal_asarray, coords2rc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RBFInterpolator
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HDLFMataLoader")

import torch
from torch.utils.data import Sampler
import random

class EventOrderedSampler(Sampler):
    """
    Shuffles events each epoch but preserves temporal order within each event.
    e.g. [event2_t0, event2_t1, event2_t2, event1_t0, event1_t1, ...]
    """
    def __init__(self, data_manager: DataManager, indices: list):
        super().__init__(data_manager.train_idx)
        self.data_manager = data_manager
        self.indices = indices
        
        # Group indices by event, preserving temporal order within each event
        self.event_groups = {}
        for idx in sorted(indices):  # sort ensures temporal order
            event_id = data_manager.find_event_id(idx)
            if event_id not in self.event_groups:
                self.event_groups[event_id] = []
            self.event_groups[event_id].append(idx)

    def __iter__(self):
        # Shuffle event order each epoch
        event_ids = list(self.event_groups.keys())
        random.shuffle(event_ids)
        
        # Yield indices event by event, in temporal order within each event
        for event_id in event_ids:
            yield from self.event_groups[event_id]

    def __len__(self):
        return len(self.indices)
    
    
class DataSet(torch.utils.data.Dataset):
    
    def __init__(self, data_manager, subset="train"):
        self.data_manager = data_manager
        self.subset = subset
        if subset == "train":
            self.indices = self.data_manager.train_idx
        else:
            raise ValueError(f"Invalid subset: {subset}. Must be 'train', 'validation', or 'test'.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        input, target = self.data_manager.get_batch(self.indices[idx], subset=self.subset)
        return input, target # Return the original index for reference

class HDLFMDataManager(DataManager):
        
    def __init__(self, config):
        super().__init__(config)
        self.dem_file = DEM_FILE
        if self.patch_domain:
            self.output_shape = (self.tile_resolution, self.tile_resolution)  # Output shape is the same as tile resolution when using tiles
        else:
            self.output_shape = self.find_study_domain_shape()
        self.input_channels = 4 if STUDY_AREA == "westport" else 5  # DEM + inundation data
        self.temporal_features = 0  # Will be updated after processing boundary conditions
        self.boundary_condition_weights = {}
        self.dem_tensor = self.map_sampler.dem_template_tiles.detach().float().cpu()  # Preprocess DEM once and store as a tensor for efficient access during training
        self.sigma = config.args.get('sigma', 100)  # Default sigma value for Gaussian smoothing
        self.find_upstream_point_fiters()

        try:
            if STUDY_AREA == "carlisle":
                self.preprocess_boundary_conditions_carlisle(self.all_event_ids)
            elif STUDY_AREA == "westport":
                self.preprocess_boundary_conditions_westport(self.all_event_ids)
            self.prepare_batch_indices()
            self.create_dataloaders(num_workers=6)
                  
        except Exception as e:
            logger.error(f"Error during data manager initialization: {e}")
            raise e
    
    def prepare_batch_indices(self):
        super().prepare_batch_indices()
        
    def create_dataloaders(self, num_workers=0):
        # Create custom samplers for train, validation, and test sets
        train_sampler = EventOrderedSampler(self, self.train_idx)

        # Create datasets
        train_dataset = DataSet(self, subset="train")

        # Create data loaders with the custom samplers
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, pin_memory_device='cuda', persistent_workers=True)

        logger.info("Data loaders created successfully with custom event-ordered samplers")

    def get_batch(self, idx, subset="train"):
        event_id = self.find_event_id(idx)
        timestep_idx, tile_group_idx = self.find_local_indices(event_id, idx)
    
        flow_tiles = self.get_flow_tensor(event_id, timestep_idx).float().to(self.device)
        output_tensor = self.get_flood_map(event_id, timestep_idx).unsqueeze(0).unsqueeze(0).float().to(self.device)
        if timestep_idx == 0:
            water_depth_tensor = self.get_flood_map(event_id, 0).unsqueeze(0).unsqueeze(0).float().to(self.device)  # Add batch and channel dimensions to water depth tensor
            # Start with zero water depth for the first timestep
        else:
            water_depth_tensor = self.get_flood_map(event_id, timestep_idx-1).unsqueeze(0).unsqueeze(0).float().to(self.device)
        water_depth_tiles = self.map_sampler.tiles_prep_func_gpu(water_depth_tensor).to(self.device)  # Add batch dimension to water depth tensor
            
        flow_tiles_for_idx = flow_tiles[tile_group_idx * self.tiles_per_index : (tile_group_idx + 1) * self.tiles_per_index].to(self.device)  # Add batch dimension to flow tensor
        dem_tiles = self.dem_tensor[tile_group_idx * self.tiles_per_index : (tile_group_idx + 1) * self.tiles_per_index].to(self.device)  # DEM tiles are preprocessed and stored as a tensor, just move to the correct device
        output_tiles = self.map_sampler.tiles_prep_func_gpu(output_tensor)[tile_group_idx * self.tiles_per_index : (tile_group_idx + 1) * self.tiles_per_index].to(self.device)  # Add batch dimension to output tensor
        water_depth_tiles = water_depth_tiles[tile_group_idx * self.tiles_per_index : (tile_group_idx + 1) * self.tiles_per_index].to(self.device)  # Add batch dimension to water depth tensor
        #should I concatanate or stack
        input_tensor = torch.cat((flow_tiles_for_idx, dem_tiles.unsqueeze(1), water_depth_tiles.unsqueeze(1)), dim=1)  # Concatenate along the channel dimension
        output_tensor = output_tiles.unsqueeze(1)  
        
        return input_tensor.cpu(), output_tensor.cpu()  # Move tensors to CPU before returning
    
    def get_test_batch(self, idx, previous_prediction):
        event_id = self.find_event_id(idx)
        timestep_idx, tile_group_idx = self.find_local_indices(event_id, idx)
        flow_tiles = self.get_flow_tensor(event_id, timestep_idx).float().to(self.device)
        if previous_prediction is None and timestep_idx == 0:
            water_depth_tensor = self.get_flood_map(event_id, 0).unsqueeze(0).unsqueeze(0).float().to(self.device)
            water_depth_tensor = self.map_sampler.tiles_prep_func_gpu(water_depth_tensor).unsqueeze(1).float().to(self.device)
            # Add batch and channel dimensions to water depth tensor
            #torch.zeros_like(self.dem_tensor).to(self.device)  # Start with zero water depth for the first timestep
        elif previous_prediction is not None:
            water_depth_tensor = previous_prediction.clamp(min=0).float().to(self.device)
            # water_depth_tensor = self.get_flood_map(event_id, timestep_idx - 1).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions to previous prediction
            #water_depth_tensor = self.map_sampler.tiles_prep_func_gpu(water_depth_tensor).to(self.device)  # Add batch dimension to water depth tensor
        else:
            raise ValueError(f"Invalid state: previous_prediction is None but timestep_idx is {timestep_idx}")
        flow_tiles_for_idx = flow_tiles[tile_group_idx * self.tiles_per_index : (tile_group_idx + 1) * self.tiles_per_index].to(self.device)  # Add batch dimension to flow tensor
        dem_tiles = self.dem_tensor[tile_group_idx * self.tiles_per_index : (tile_group_idx + 1) * self.tiles_per_index].to(self.device) 
        water_depth_tiles = water_depth_tensor[tile_group_idx * self.tiles_per_index : (tile_group_idx + 1) * self.tiles_per_index].to(self.device)  # DEM tiles are preprocessed and stored as a tensor, just move to the correct device
        input_tensor = torch.cat((flow_tiles_for_idx, dem_tiles.unsqueeze(1), water_depth_tiles), dim=1) 
        return input_tensor.float(), None

    def get_flow_tensor(self, event_id, timestep_idx):
        event_data = self.event_input_map[event_id][timestep_idx]

        T, H, W = self.dem_tensor.shape
        flow_tensors = []

        for i in range(self.temporal_features):
            # Scale spatial weight map by the scalar gauge reading at this timestep
            flow_tensor = torch.ones((T, H, W)) * torch.tensor(event_data[i], dtype=torch.float32)                            # scalar broadcast
            flow_tensor = flow_tensor * self.boundary_condition_weights[f"bc{i+1}"] 
            flow_tensors.append(flow_tensor)

        #Now need to stack the flow tensors along the channel dimension to create a multi-channel input for the model. The resulting shape will be (features, T, H, W) where features is the number of temporal features (e.g. 3 for carlisle, 5 for westport).
        concatanated_flow_tensors = torch.stack(flow_tensors, dim=0)
        #swap the T and features dimensions to get the final shape of (T, features, H, W) which is what the model expects as input
        concatanated_flow_tensors = concatanated_flow_tensors.permute(1, 0, 2, 3)
        return concatanated_flow_tensors.float()

    # def create_boundary_weights(self, coordinates):
    #     sample_file = self.find_sample_flood_map()
    #     square_size = 7
    #     with rio.open(sample_file) as src:
    #         transform = src.transform
    #         row, column = coords2rc(transform, (coordinates['easting'], coordinates['northing']))
    #         square = {
    #             'row_start': row - square_size // 2,
    #             'row_end': row + square_size // 2,
    #             'col_start': column - square_size // 2,
    #             'col_end': column + square_size // 2
    #         }
        
    #     upstream_map = torch.zeros_like(self.dem_tensor)
    #     upstream_map[square['row_start']:square['row_end'], square['col_start']:square['col_end']] = 1
    #     upstream_filter = upstream_map > 0
    #     return upstream_filter
    
    def interpolate_boundary_condition(self, coordinates):
        sample_file = self.find_sample_flood_map()
        
        with rio.open(sample_file) as src:
            transform = src.transform
            height = src.height
            width = src.width

        cols, rows = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )

        xs, ys = rio.transform.xy(transform, rows, cols)
        xs = torch.tensor(xs, dtype=torch.float32)
        ys = torch.tensor(ys, dtype=torch.float32)

        dist_sq = (
            (xs - coordinates["easting"]) ** 2 +
            (ys - coordinates["northing"]) ** 2
        )

        weights = torch.exp(-dist_sq / (2 * self.sigma**2))
        #reshape to match the spatial dimensions of the input tensor
        weights = weights.reshape((height, width))
        return weights

    def find_upstream_point_fiters(self):
        if STUDY_AREA == "carlisle":
            # To create the 2D input from the hydrograph
            bc_coordinates = [
                {"name": "bc1", "easting": 342682, "northing": 557532},
                {"name": "bc2", "easting": 341362, "northing": 554702},
                {"name": "bc3", "easting": 339947, "northing": 554702},
            ]
                
        elif STUDY_AREA == "westport":
            bc_coordinates = [
                {"name": "bc1", "easting": 1480982.0, "northing": 5381961.2},
                {"name": "bc2", "easting": 1479682.0, "northing": 5379571.2},
            ]
        
        # Above are main boundary conditions for study areas. 
        # Now if tiles are used when training the models. Needs to assign the bcs to tiles. 
        # So need to create an artifical boundary condition for each tile based on the distance to the main boundary condition. This is done by creating a filter for each tile that assigns the value of the main boundary condition to the tile if it is within a certain distance.
        # Create a weighted filter for each cell in the flood map based on the distance to the main boundary condition. The weight is calculated as 1/(distance + 1) to avoid division by zero. This way, cells closer to the boundary condition will have a higher weight and contribute more to the input tensor.
        
        for coordinates in bc_coordinates:
            filter = self.interpolate_boundary_condition(coordinates)  
            if self.patch_domain: 
                filter = self.map_sampler.tiles_prep_func_gpu(filter.unsqueeze(0).unsqueeze(0)).detach().cpu()  # Preprocess the filter to match the tile preparation function 
            self.boundary_condition_weights[coordinates['name']] =  filter
    
    def preprocess_boundary_conditions_carlisle(self, event_ids):
        self.event_input_map = {}
        flow_data_map = {}
        timestep_data_map = {}

        for event_id in event_ids:
            inflow_file = os.path.join(BC_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            event_index = self.event_index_map[event_id]
              
            flood_map_files = glob.glob(os.path.join(FLOOD_MAPS_DIR, f"{self.event_name_map[event_index]}", "Run*.wd"))
            flood_map_files.sort()
            
            # Extract the timestamp from the flood map file names and store in a list
            flood_map_timestamps = [self.extract_timestep_from_filename(file) for file in flood_map_files]
            logger.info(f"Extracted {len(flood_map_timestamps)} flood map   timestamps for event ID: {event_id}")
            inflow_data['timestep'] = (inflow_data['Time'].astype(np.float32) / (60 * 15)).astype(np.float32)
            # Use only a single timestep column (no lagged timestep-* columns)
            time_columns = ['timestep']
            inflow_data['flood_map_file'] = inflow_data['timestep'].apply(lambda x: next((file for file in flood_map_files if int(self.extract_timestep_from_filename(file)) == int(x)), None))

            # Remove first 8 rows as first 8 timesteps represent the intialisation period. 
            inflow_data = inflow_data.iloc[8:]
            inflow_data = inflow_data.dropna()
            inflow_data = inflow_data.drop(columns=['Time'])
            self.flood_map_file_map[event_id] = inflow_data['flood_map_file'].values #Store flood map file paths for each event

            flow_columns = [col for col in inflow_data.columns if 'Upstream' in col and 'timestep' not in col]
            flow_data_map[event_id] = inflow_data[flow_columns].values.astype(np.float32)
            timestep_data_map[event_id] = inflow_data['timestep'].values.astype(np.float32)
            
        flow_train_data = np.vstack([flow_data_map[event_id] for event_id in self.train_event_ids])
        flow_scaler = MinMaxScaler(feature_range=(0, 1))
        flow_scaler.fit(flow_train_data)
        self.temporal_features = len(flow_columns)  # Number of features is the number of flow columns plus one timestep column

        for event_id in self.all_event_ids:
            scaled_flow = flow_scaler.transform(flow_data_map[event_id]).astype(np.float32)
            time_data = timestep_data_map[event_id]

            self.event_input_map[event_id] = np.hstack((scaled_flow, time_data.reshape(-1, 1)))
            logger.info(f"Preprocessed boundary condition data for event ID: {event_id} with shape {self.event_input_map[event_id].shape}")
        # self.temporal_features = self.event_input_map[self.all_event_ids[0]].shape[1]  # Number of features is the second dimension of the input data
        
        logger.info(f"Updated feature count: {self.temporal_features} (including timestep history)")
        
        
    def preprocess_boundary_conditions_westport(self, event_ids):
        self.unscaled_inputs = {}
        self.flood_map_file_map = {} # Store flood map file paths for each event
        self.event_input_map = {}
        for event_id in event_ids:
            event_index = self.event_index_map[event_id]
            logger.info(f"Preprocessing boundary conditions for event ID: {event_id}, Event Name: {self.event_name_map[event_index]} Event Index: {event_index}")
            inflow_file = os.path.join(BC_DATA_DIR, f"{event_index}.csv")
            inflow_data = pd.read_csv(inflow_file)
            
            # Drop first column as it is just an index column
            inflow_data = inflow_data.drop(inflow_data.columns[0], axis=1)
            
            flood_map_files = glob.glob(os.path.join(FLOOD_MAPS_DIR, f"{self.event_name_map[event_index]}_depth_snapshots", "depth_at_*.tif"))
            flood_map_files.sort()
            
            # Extract the timestamp from the flood map file names and store in a list
            flood_map_timestamps = [self.extract_timestep_from_filename(file) for file in flood_map_files]
            
            
            #Add a column to the inflow data for the corresponding flood map timestamp. If not found add a default value.
            inflow_data['Date_Time'] = pd.to_datetime(inflow_data['Date_Time'])
            inflow_data['Flood_Map_Available'] = inflow_data['Date_Time'].apply(lambda x: True if x in flood_map_timestamps else False)
            inflow_data['flood_map_file'] = inflow_data['Date_Time'].apply(lambda x: next((file for file in flood_map_files if self.extract_timestep_from_filename(file) == x), None))
            
            inflow_data.dropna(subset=['flood_map_file'], inplace=True)
            
            date_time = inflow_data["Date_Time"]
            date_time = pd.to_datetime(date_time)
            timestep = (date_time - date_time.min()).dt.total_seconds() / 3600.0
            inflow_data['timestep'] = timestep.values.astype(np.float32)  # single timestep column only  
            inflow_data = inflow_data.dropna()
        
            # Make timestep start from 0 for each time-step column and drop the original Date_Time column
            # Make timestep start from 0 (single column)
            inflow_data['timestep'] = inflow_data['timestep'] - inflow_data['timestep'].min()
            self.flood_map_file_map[event_id] = inflow_data['flood_map_file'].values # Store flood map file paths for each event
            
            # Drop the original Date_Time column and convert the rest to a numpy array
            inflow_data.drop(columns=['Flood_Map_Available', 'flood_map_file', 'Date_Time'], inplace=True) 
            # Store input data for this event
            self.unscaled_inputs[event_id] = inflow_data
            logger.info(f"Extracted {len(inflow_data)} valid timesteps with flood maps for event ID: {event_id}")
            
        # Fit a scaler on all training data
        flow_columns = [col for col in self.unscaled_inputs[event_id].columns if 'River_flow' in col]
        flow_train_data = np.vstack([self.unscaled_inputs[id][flow_columns].values for id in self.train_event_ids])
        flow_scaler = MinMaxScaler(feature_range=(0, 1))
        flow_scaler.fit(flow_train_data)
        
        sea_level_columns = [col for col in self.unscaled_inputs[event_id].columns if 'Sea_level' in col]
        sea_level_train_data = np.vstack([self.unscaled_inputs[id][sea_level_columns].values for id in self.train_event_ids])
        sea_level_scaler = MinMaxScaler(feature_range=(0, 1))
        sea_level_scaler.fit(sea_level_train_data)
        self.temporal_features = len(flow_columns) + len(sea_level_columns)  # Number of features excluding timestep data

        for event_id in self.all_event_ids:
            scaled_flow = flow_scaler.transform(self.unscaled_inputs[event_id][flow_columns].values)
            # Convert the values to float32 to save memory and ensure compatibility with PyTorch
            scaled_flow = scaled_flow.astype(np.float32)
            
            scaled_sea_level = sea_level_scaler.transform(self.unscaled_inputs[event_id][sea_level_columns].values)
            scaled_sea_level = scaled_sea_level.astype(np.float32)
            
            time_step_columns = [col for col in self.unscaled_inputs[event_id].columns if 'timestep' in col]
            time_step_data = self.unscaled_inputs[event_id][time_step_columns].values.astype(np.float32)
            # Don't scale time data 
            # scaled_time = time_scaler.transform(self.timestamp_map[event_id])
            # scaled_time = scaled_time.astype(np.float32)

            # Combine time and flow data
            # Replace with scaled data in the event input map
            # self.event_input_map[event_id] = np.hstack((self.timestamp_map[event_id], scaled_flow, scaled_sea_level))
            
            # Convert to # Shape: (time_steps, features, lag)
            lagged_flow_sea_level_time = np.hstack((scaled_flow, scaled_sea_level, time_step_data)) 
            logger.info(f"Preprocessed boundary condition data for event ID: {event_id} with shape {lagged_flow_sea_level_time.shape}")
            
            self.event_input_map[event_id] = lagged_flow_sea_level_time
            
        # self.temporal_features = self.event_input_map[self.all_event_ids[0]].shape[1]
        logger.info(f"Preprocessed boundary condition data for events: {event_ids}")
     
            
    # def inflow_data_prep(self, event_ids):
    #     self.event_input_map = {}
    #     self.event_data_map_unscaled = {}
    #     self.time_data_map = {}  # Store time data separately
        
    #     for idx in range(len(event_ids)):
    #         event_id = event_ids[idx]
    #         inflow_file = os.path.join(BC_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
    #         inflow_data = pd.read_csv(inflow_file)
    #         inflow_data = inflow_data[8:] #Skip the first 8 rows
    #         inflow_data = inflow_data.dropna()
    #         flow_data = inflow_data.drop(columns=['Time'])
    #         input_arr = np.array(flow_data.values.astype(np.float32))
    #         self.event_input_map[event_id] =  input_arr
               
    #     #Fit the flow scaler on all training data
    #     all_train_data = np.vstack([self.event_input_map[id] for id in self.train_event_ids])
    #     flow_scaler = MinMaxScaler(feature_range=(0, 1))
    #     flow_scaler.fit(all_train_data)
        
    #     for event_id in self.all_event_ids:
    #         scaled_flow = flow_scaler.transform(self.event_input_map[event_id])
    #         self.event_input_map[event_id] = scaled_flow                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    