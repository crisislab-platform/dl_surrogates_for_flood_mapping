import logging
import os
import numpy as np
import pandas as pd
import rasterio as rio
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from modules.lib.constants import BC_DATA_DIR, SIMULATION_DATA_DIR, DEM_FILE, FLOOD_MAPS_DIR, STUDY_AREA
from modules.datamanager.datamanager import DataManager
from modules.utils.run_util import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataLoader")

elevation_file_path = DEM_FILE
bc_data_dir = BC_DATA_DIR
lisflood_simulation_dir = SIMULATION_DATA_DIR

class CNN1DDataManager(DataManager):
    
    def __init__(self, config):
        super().__init__(config)
        self.lag = config.lag
        # Calculate features: 3 upstream flows and timestep, each with lagged history
        self.input_features = (self.lag + 1) * 4
        self.input_features = None
        self.output_shape = self.find_study_domain_shape()
        self.outputs = self.output_shape[0] * self.output_shape[1]
        
        self.event_input_map = {}
        self.inundation_data_cache = {}
        
        try:
            if STUDY_AREA == "carlisle":
                self.preprocess_boundary_conditions_carlisle(self.all_event_ids)
            elif STUDY_AREA == "westport":
                self.preprocess_boundary_conditions_westport(self.all_event_ids)
            self.prepare_batch_indices()
            self.create_dataloaders(num_workers=2)
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            # Set defaults to avoid None errors
            raise e
        
    def prepare_batch_indices(self):
        self.event_start_id_map = {}
        # Calculate start indices for each event
        for event_id in self.all_event_ids:
            if event_id == 1:  # First event
                start_idx = 0
            else:
                prev_event_id = event_id - 1
                start_idx = self.event_start_id_map[prev_event_id] + self.get_no_time_steps(prev_event_id) * self.indices_per_timestep
            self.event_start_id_map[event_id] = start_idx
  
        self.train_idx = self.index_expander(self.train_event_ids)
        self.validation_idx = self.index_expander(self.validation_event_ids)
        
        self.train_and_val_indices = np.concatenate((self.train_idx, self.validation_idx))
        #70% for training and 30% for validation from the combined train and val indices
        self.train_idx = np.random.choice(self.train_and_val_indices, size=int(0.7 * len(self.train_and_val_indices)), replace=False)
        self.validation_idx = np.setdiff1d(self.train_and_val_indices, self.train_idx)
        self.test_idx = self.index_expander(self.test_event_ids)
            
   
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
            
            for i in range(1, self.lag + 1):
                inflow_data[f'Upstream1-{i}'] = inflow_data['Upstream1'].shift(i)
                inflow_data[f'Upstream2-{i}'] = inflow_data['Upstream2'].shift(i)
                inflow_data[f'Upstream3-{i}'] = inflow_data['Upstream3'].shift(i)

            inflow_data = inflow_data.dropna()
            inflow_data = inflow_data.drop(columns=['Time'])
            self.flood_map_file_map[event_id] = inflow_data['flood_map_file'].values # Store flood map file paths for each event

            flow_columns = [col for col in inflow_data.columns if 'Upstream' in col and 'timestep' not in col]
            flow_data_map[event_id] = inflow_data[flow_columns].values.astype(np.float32)
            timestep_data_map[event_id] = inflow_data['timestep'].values.astype(np.float32)
            
        flow_train_data = np.vstack([flow_data_map[event_id] for event_id in self.train_event_ids])
        flow_scaler = MinMaxScaler(feature_range=(0, 1))
        flow_scaler.fit(flow_train_data)

        for event_id in self.all_event_ids:
            scaled_flow = flow_scaler.transform(flow_data_map[event_id]).astype(np.float32)
            time_data = timestep_data_map[event_id]

            self.event_input_map[event_id] = np.hstack((scaled_flow, time_data.reshape(-1, 1)))
            logger.info(f"Preprocessed boundary condition data for event ID: {event_id} with shape {self.event_input_map[event_id].shape}")
        self.input_features = self.event_input_map[self.all_event_ids[0]].shape[1]  # Number of features is the second dimension of the input data
        logger.info(f"Updated feature count: {self.input_features} (including timestep history)")
        
        
    def preprocess_boundary_conditions_westport(self, event_ids):
        self.unscaled_inputs = {}
        self.flood_map_file_map = {} # Store flood map file paths for each event
        
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

            #Create lagged features for flow and sea level only (do NOT create lagged timestep columns)
            for i in range(1, self.lag + 1):
                inflow_data[f'River_flow-{i}'] = inflow_data['River_flow'].shift(i)
                inflow_data[f'Sea_level-{i}'] = inflow_data['Sea_level'].shift(i)
                
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
            
        self.input_features = self.event_input_map[self.all_event_ids[0]].shape[1]
        logger.info(f"Preprocessed boundary condition data for events: {event_ids}")

    
    def get_batch(self, idx, subset="train"):
        event_id = self.find_event_id(idx)
        timestep, _ = self.find_local_indices(event_id, idx)
        event_input = torch.from_numpy(self.event_input_map[event_id][timestep]).float()
        flood_map_file = self.flood_map_file_map[event_id][timestep]
        flood_map_tensor = self.load_inundation_data(event_id=event_id, files=[flood_map_file])[0]
        event_input = event_input.unsqueeze(0) 
        flood_map_tensor = flood_map_tensor.unsqueeze(0)  # Add channel dimension
        return event_input, flood_map_tensor
