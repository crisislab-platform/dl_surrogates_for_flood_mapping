import logging
import os
import numpy as np
import pandas as pd
import rasterio as rio
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from modules.lib.constants import CARLISLE_DATA_DIR, SIMULATION_DATA_DIR
from modules.datamanager.datamanager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataLoader")

elevation_file_path = f'{CARLISLE_DATA_DIR}/Carlisle_5m.asc'
bc_data_dir = f"{CARLISLE_DATA_DIR}"
lisflood_simulation_dir = SIMULATION_DATA_DIR

class CNNRasterDataManager(DataManager):
    def __init__(self, lag=8, batch_size=32, device='cpu', pinn=False, validation_event=2, tuning_mode=True):
        if tuning_mode:
            self.train_event_ids = [event for event in [2, 3, 4, 5, 6, 7, 8, 9] if event != validation_event]
            self.test_event_ids = [1]
            self.val_event_ids = [validation_event]
        else:   
            self.train_event_ids = [2, 3, 4, 5, 6, 7, 8, 9]
            self.test_event_ids = [1]
            self.val_event_ids = [1]
            
        self.all_event_ids = np.concatenate([self.train_event_ids, self.test_event_ids, self.val_event_ids])
        self.all_event_ids.sort()
        self.timestep = 1 # One 15 min time step
        self.lag = lag # No of antecedent time steps
        self.input_seq_length = lag
        self.up1_baseline_fr = 69.2354
        self.up2_baseline_fr = 6.2953
        self.up3_baseline_fr = 14.7835
        self.batch_size = batch_size
        self.train_idx = None
        self.validation_idx = None
        self.device = device  # Store device for tensor operations
        # Calculate features: 3 upstream flows × lag steps + original 3 upstream flows + time column
        self.features = (lag * 3) + 3
        self.outputs = self.find_output_size()
        self.pinn = pinn
        self.event_data_map = {}
        self.event_batch_map = {}
        
        # Initialize with proper error handling
        try:
            self.inflow_data_prep(self.all_event_ids)
            self.prep_test_ouput_data()
            self.idx_prep()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            # Set defaults to avoid None errors
            raise e
            
    def find_output_size(self):
        # Load a sample inundation file to determine the output size
        sample_file = f"{lisflood_simulation_dir}/Run1-0000.wd"
        sample_data = self.process_inundation_file(sample_file)
        return sample_data.shape[0]
    
    def get_no_time_steps(self, event_id):
        inundation_files = glob.glob(f"{lisflood_simulation_dir}/Run{event_id}-*.wd")
        return len(inundation_files) - self.lag #Exclude the first 8 files and lag files
    
    def load_inundation_data(self, event_id, time_indices):
        event_inundation_files = glob.glob(f"{lisflood_simulation_dir}/Run{event_id}-*.wd")
        event_inundation_files.sort()
        event_inundation_files = event_inundation_files[self.lag:] #Exclude the first #lag files
        
        event_inundation_files = [event_inundation_files[i] for i in time_indices]
        event_inundation_data = []
        for i, file in enumerate(event_inundation_files):
            inundation_data = self.process_inundation_file(file)
            event_inundation_data.append(inundation_data)
        event_inundation_data = np.array(event_inundation_data)
        # Set negligible inundation to 0
        event_inundation_data[event_inundation_data < 0.3] = 0
        return torch.from_numpy(event_inundation_data).float()

    def inflow_data_prep(self, event_ids, test_mode=False):
        self.event_data_map = {}
        self.event_data_map_unscaled = {}
        for idx in range(len(event_ids)):
            event_id = event_ids[idx]
            inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            
            # Create lagged features for each upstream input
            for i in range(1, self.lag + 1):
                inflow_data[f'Upstream1-{i}'] = inflow_data['Upstream1'].shift(i)
                inflow_data[f'Upstream2-{i}'] = inflow_data['Upstream2'].shift(i)
                inflow_data[f'Upstream3-{i}'] = inflow_data['Upstream3'].shift(i)
                 
            inflow_data = inflow_data.dropna()
            # inflow_data["Time"] = inflow_data["Time"].astype(float) / 3600 #convert to hours
            inflow_data = inflow_data.drop(columns=['Time'])
            input_arr = np.array(inflow_data.values.astype(np.float32))
            #get first 3 columns
            self.event_data_map_unscaled[event_id] = input_arr[:, :3]
            self.event_data_map[event_id] = input_arr
               
        # Fit the scaler on each event's data 
        self.event_data_map[event_id] = input_arr
        all_train_data = np.vstack([self.event_data_map[id] for id in self.train_event_ids])
        flow_scaler = MinMaxScaler(feature_range=(0, 1))
        flow_scaler.fit(all_train_data)

        # Then transform each event
        for event_id in self.all_event_ids:
            self.event_data_map[event_id] = flow_scaler.transform(self.event_data_map[event_id])
            if event_id in self.test_event_ids:
                input_arr = self.event_data_map[event_id]
                input_arr = input_arr.reshape(input_arr.shape[0], 1, input_arr.shape[1])
                self.test_input = torch.from_numpy(input_arr)
            
    def process_inundation_file(self, file_path):
        try:
            with rio.open(file_path) as src:
                data = src.read(1)
                return data.flatten()
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            return None
    
    def index_expander(self, x:list):
        idx_batches = []
        logger.info(f"Expanding indices for {len(x)} events")
        
        for event_id in x:
            start_idx = self.event_start_id_map[event_id]
            num_idx = self.timesteps_per_event(event_id)
            
            # Calculate number of possible full batches based on available timesteps
            # rather than using potentially inconsistent pre-computed batch counts
            num_full_batches = num_idx // self.batch_size
                
            # Use the minimum of pre-calculated batch count and actually possible batches
            # to avoid creating indices beyond available data
            event_batches = []
            
            for i in range(num_full_batches):
                batch_start = i * self.batch_size
                batch_end = min((i + 1) * self.batch_size, num_idx)
                if batch_end - batch_start < self.batch_size:
                    # Skip incomplete batches
                    continue
                    
                # Create actual indices for this batch relative to the start of this event
                batch_indices = start_idx + np.arange(batch_start, batch_end)
                idx_batches.append(batch_indices)
                event_batches.append(i)  # Store batch index within this event
                
            # Store mapping of event to its batch indices
            self.event_batch_map[event_id] = event_batches

        if len(idx_batches) > 0:
            idx_batches = np.array(idx_batches)
            logger.info(f"Index batches shape: {idx_batches.shape}")
            return idx_batches
        else:
            logger.error("No batches were created! Check your data and batch size.")
            return np.array([])  #Return empty array as fallback
    
    def find_event_id(self, idx):
        for event_id, start_idx in self.event_start_id_map.items():
            end_idx = start_idx + self.timesteps_per_event(event_id)
            if start_idx <= idx and idx < end_idx:
                return event_id
        return None
    
    def find_time_steps(self, idxs):
        event_id = self.find_event_id(idxs[0])
        if event_id is None:
            logger.error(f"Could not find event ID for index {idxs[0]}")
            return None, np.array([])
            
        start_idx = self.event_start_id_map[event_id]
        # Convert global indices to event-local indices
        local_idxs = [idx - start_idx for idx in idxs]
        return event_id, np.array(local_idxs)
        
    def idx_prep(self, shuffle=True, random_seed=341):
        logger.info(f"Preparing index for batch size: {self.batch_size}")
        self.timesteps_per_event = lambda y: np.ceil((self.get_no_time_steps(y) - self.timestep//2) / self.timestep).astype(int)
        self.event_start_id_map = {}
        
        # Calculate start indices for each event
        for event_id in self.all_event_ids:
            if event_id == 1:  # First event
                start_idx = 0
            else:
                prev_event_id = event_id - 1
                start_idx = self.event_start_id_map[prev_event_id] + self.timesteps_per_event(prev_event_id)
            self.event_start_id_map[event_id] = start_idx
        
        try:
            train_idx_batches = self.index_expander(self.train_event_ids)
            validation_idx_batches = self.index_expander(self.val_event_ids)
            
            # Check if batches were created successfully
            if len(train_idx_batches) == 0 or len(validation_idx_batches) == 0:
                logger.error("Failed to create index batches")
                return
                
            if shuffle:
                rng = np.random.default_rng(random_seed)
                # Shuffle along the first dimension (batches)
                rng.shuffle(train_idx_batches)
                    
            self.train_idx = train_idx_batches
            self.validation_idx = validation_idx_batches
            
            # Store the number of batches for logging
            self.train_batches = len(train_idx_batches)
            self.val_batches = len(validation_idx_batches)
            logger.info(f"Created {self.train_batches} training batches and {self.val_batches} validation batches")
            
        except Exception as e:
            logger.error(f"Error in idx_prep: {e}")
            # Initialize with empty arrays as fallback
            self.train_idx = np.array([])
            self.validation_idx = np.array([])
            self.train_batches = 0
            self.val_batches = 0
        
    def get_batch(self, indices):
        if indices is None or len(indices) == 0:
            logger.error("Empty batch indices passed to get_batch")
            return None
        
        # Find event ID and local indices
        event_id, local_indices = self.find_time_steps(indices)
        if event_id is None:
            logger.error("Could not determine event ID for batch")
            return None
        logger.info(f"Getting batch for event ID {event_id} with {len(local_indices)} indices")
        
        try:
            # Get inflow data for this batch
            if event_id not in self.event_data_map:
                logger.error(f"No data available for event {event_id}")
                return None, None
                
            # Get the inflow data for these specific indices
            event_data = self.event_data_map[event_id]
            batch_data = np.array([event_data[idx] for idx in local_indices if idx < len(event_data)])
            batch_data = batch_data.reshape(batch_data.shape[0], 1, batch_data.shape[1])
            
            # Convert to tensor
            input_batch = torch.from_numpy(batch_data).float()
            output_batch = self.load_inundation_data(event_id, local_indices)
            
            if self.pinn:
                # Since we don't use the complete timesteps as the number of timesteps of any event is not devisible by the batch size,
                # we can  assume idx+1 is always idx + 1 < self.get_no_time_steps(event_id)
                next_indices = [idx + 1 for idx in local_indices if idx + 1 < self.get_no_time_steps(event_id)]
                if next_indices:
                    yt_plus1 = self.load_inundation_data(event_id, next_indices)
                else:
                    yt_plus1 = None
            
                bct = torch.from_numpy(self.get_boundary_conditions(local_indices))
                bct_plus1 = torch.from_numpy(self.get_boundary_conditions(next_indices) if next_indices else None)
                return input_batch, output_batch, yt_plus1, bct, bct_plus1
            else:
                return input_batch, output_batch
            
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            raise e

    def get_boundary_conditions(self, indices):
        bc_data = np.zeros((len(indices), 3))
    
        for i, idx in enumerate(indices):
            event_id, local_idx = self.find_time_steps([idx])
            if event_id is None:
                logger.error(f"Could not find event ID for index {idx}")
                continue
            
            event_data = self.event_data_map_unscaled[event_id]
            for idx in local_idx:
                if idx < len(event_data):
                    bc_data[i, 0] = event_data[idx, 0]
                    bc_data[i, 1] = event_data[idx, 1]
                    bc_data[i, 2] = event_data[idx, 2]
                else:
                    bc_data[i, 0] = self.up1_baseline_fr
                    bc_data[i, 1] = self.up2_baseline_fr
                    bc_data[i, 2] = self.up3_baseline_fr
        return bc_data

    def prep_test_ouput_data(self):
        time_steps = self.get_no_time_steps(self.test_event_ids[0])
        time_indices = np.arange(time_steps)
        inundation_data = self.load_inundation_data(self.test_event_ids[0], time_indices)
        self.test_output= inundation_data
        logger.info(f"Test output data shape: {self.test_output.shape}")