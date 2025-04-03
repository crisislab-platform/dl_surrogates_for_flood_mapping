from modules.lib.constants import PROJECT_ROOT
import logging
import os
import numpy as np
import pandas as pd
import rasterio as rio
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from modules.lib.constants import CARLISLE_DATA_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_asarray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataLoader")

elevation_file_path = f'{CARLISLE_DATA_DIR}/Carlisle_5m.asc'
bc_data_dir = f"{CARLISLE_DATA_DIR}"
lisflood_simulation_dir = SIMULATION_DATA_DIR


TEST_SUBSET = "Run1" #Run1 is the test data corresponding to the 2005 event real streamflow data
VALIDATION_SUBSET = "Run9" #Use Run9 for validation in the training process
# Train on Run2-Run8 data (7 design events)

class CNNRasterDataManager():
    def __init__(self, lag=8, batch_size=32, device='cpu', test_mode=False):
        self.train_event_ids = [2, 3, 4, 5, 6, 7, 8]
        self.test_event_ids = [1]
        self.val_event_ids = [9]
        self.all_event_ids = np.concatenate([self.train_event_ids, self.test_event_ids, self.val_event_ids])
        self.all_event_ids.sort()
        self.timestep = 1 # one 15 min time step
        self.lag = lag # no of antecedent time steps
        self.input_seq_length = lag
        self.up1_baseline_fr = 69.2354
        self.up2_baseline_fr = 6.2953
        self.up3_baseline_fr = 14.7835
        self.batch_size = batch_size
        self.train_idx = None
        self.validation_idx = None
        self.device = device  # Store device for tensor operations
        self.features = lag * 3 + 3  # 3 upstream flows and lagged features for each
        self.outputs = self.find_output_size()
        
        # Initialize with proper error handling
        try:
            if test_mode:
                self.inflow_data_prep(self.test_event_ids, True) 
                self.prep_test_ouput_data()
                return
            self.inflow_data_prep(self.all_event_ids)
            self.idx_prep()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            # Set defaults to avoid None errors
            self.input_data = torch.zeros((1, self.batch_size, self.input_seq_length), device=self.device)
            
    def find_output_size(self):
        # Load a sample inundation file to determine the output size
        sample_file = f"{lisflood_simulation_dir}/Run1-0000.wd"
        sample_data = self.process_inundation_file(sample_file)
        return sample_data.shape[0]
    
    def get_no_time_steps(self, event_id):
        inundation_files = glob.glob(f"{lisflood_simulation_dir}/Run{event_id}-*.wd")
        return len(inundation_files) - 8
    
    def load_inundation_data(self, event_id, time_indices):
        event_inundation_files = glob.glob(f"{lisflood_simulation_dir}/Run{event_id}-*.wd")
        event_inundation_files.sort()
        event_inundation_files = event_inundation_files[8:]
        
        event_inundation_files = [event_inundation_files[i] for i in time_indices]
        event_inundation_data = []
        for i, file in enumerate(event_inundation_files):
            inundation_data = self.process_inundation_file(file)
            event_inundation_data.append(inundation_data)
        event_inundation_data = np.array(event_inundation_data)
        event_inundation_data[event_inundation_data < 0.3] = 0
        return torch.from_numpy(event_inundation_data).float()

    
    def inflow_data_prep(self, event_ids, test_mode=False):
        # Modified to use lagged features instead of sequences
        self.event_data_map = {}  # Store data for each event
        event_input = []
        
        for event_id in event_ids:
            inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            inflow_data = inflow_data.iloc[8:]
            inflow_data = inflow_data.reset_index(drop=True)
            inflow_data = inflow_data.drop(columns=['Time'])
            # Create lagged features for each upstream input
            for i in range(1, self.lag+1):
                inflow_data[f'Upstream1-{i}'] = inflow_data['Upstream1'].shift(i)
                inflow_data[f'Upstream2-{i}'] = inflow_data['Upstream2'].shift(i)
                inflow_data[f'Upstream3-{i}'] = inflow_data['Upstream3'].shift(i)
            
            inflow_data = inflow_data.dropna()
            logger.info(f"Event {event_id} inflow data shape: {inflow_data.shape}")
            self.event_data_map[event_id] = inflow_data.values
            input_arr = np.array(inflow_data)
            num_rows = input_arr.shape[0]
            if not test_mode:
                num_full_batches = num_rows // self.batch_size
                if not hasattr(self, 'event_batch_counts'):
                    self.event_batch_counts = {}
                self.event_batch_counts[event_id] = num_full_batches
                
                for i in range(num_full_batches):
                    batch_start = i * self.batch_size
                    batch_end = (i + 1) * self.batch_size
                    batch = input_arr[batch_start:batch_end]
                    batch = batch.reshape(batch.shape[0], 1, batch.shape[1])
                    event_input.append(batch)
            else:
                # For test mode, just append the entire event data
                batch = input_arr.reshape(num_rows, 1, input_arr.shape[1])
                event_input.append(batch)
        
        if  test_mode:
            event_input = np.concatenate(event_input, axis=0)
            logger.info(f"Test event input shape: {event_input.shape}")
        
        if len(event_input) > 0:
            event_input = np.array(event_input)
            logger.info(f"Event input shape: {event_input.shape}")
            self.input_data = torch.from_numpy(event_input).float().to(self.device)
        else:
            logger.error("No input data was created! Check your data and batch size.")
            self.input_data = torch.tensor([]).to(self.device)
    
    def process_inundation_file(self, file_path):
        """Process a single inundation file and return its data as a flattened array."""
        try:
            with rio.open(file_path) as src:
                data = src.read(1)
                return data.flatten()
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            # Return zeros array with appropriate shape as fallback
            return None  # Assuming this is the standard size
    
    def index_expander(self, x:list):
        idx_batches = []
        self.event_batch_map = {}  # Maps event_id to its batch indices
        logger.info(f"Expanding indices for {len(x)} events")
        
        for event_id in x:
            start_idx = self.event_start_id_map[event_id]
            num_idx = self.timesteps_per_event(event_id)
            if event_id not in self.event_batch_counts:
                logger.warning(f"No batch count found for event {event_id}, skipping")
                continue
                
            num_full_batches = self.event_batch_counts[event_id]
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
            return np.array([])  # Return empty array as fallback
    
    def find_event_id(self, idx):
        """Find which event an index belongs to."""
        for event_id, start_idx in self.event_start_id_map.items():
            end_idx = start_idx + self.timesteps_per_event(event_id)
            if start_idx <= idx and idx < end_idx:
                return event_id
        return None
    
    def find_time_steps(self, idxs):
        """Convert global indices to event-specific indices."""
        # Assuming all indices in the batch belong to the same event
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

        # Make sure we have event batch counts before calling this
        if not hasattr(self, 'event_batch_counts'):
            logger.error("Event batch counts not available. Call inflow_data_prep first.")
            self.event_batch_counts = {event_id: 0 for event_id in self.all_event_ids}

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
                rng.shuffle(validation_idx_batches)
                    
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
        
    def get_batch(self, batch_indices):
        if batch_indices is None or len(batch_indices) == 0:
            logger.error("Empty batch indices passed to get_batch")
            return None
        
        # Find event ID and local indices
        event_id, local_indices = self.find_time_steps(batch_indices)
        if event_id is None:
            logger.error("Could not determine event ID for batch")
            return None
            
        logger.info(f"Getting batch for event ID {event_id} with {len(local_indices)} indices")
        
        try:
            # Get inflow data for this batch
            if event_id not in self.event_data_map:
                logger.error(f"No data available for event {event_id}")
                return None,None
                
            # Get the inflow data for these specific indices
            event_data = self.event_data_map[event_id]
            batch_data = np.array([event_data[idx] for idx in local_indices if idx < len(event_data)])
            
            # Reshape to expected format (batch_size, 1, features)
            batch_data = batch_data.reshape(batch_data.shape[0], 1, batch_data.shape[1])
            
            # Convert to tensor
            x_batch = torch.from_numpy(batch_data).float().to(self.device)
            
            # Get inundation data (target)
            inundation_data = self.load_inundation_data(event_id, local_indices)
            y_batch = inundation_data.to(self.device)
            return x_batch, y_batch
            
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            return torch.zeros((self.batch_size, 1, self.input_seq_length * 3), device=self.device), torch.zeros((self.batch_size, 581061), device=self.device)

    def prep_test_ouput_data(self):
        # Load inundation data for the test event
        time_steps = self.get_no_time_steps(self.test_event_ids[0])
        time_indices = np.arange(time_steps)
        time_indices = time_indices[8:]  # Skip the first 8 time steps
        inundation_data = self.load_inundation_data(self.test_event_ids[0], time_indices)
        self.test_output= inundation_data
        self.test_output = self.test_output.to(self.device)
        logger.info(f"Test output data shape: {self.test_output.shape}")