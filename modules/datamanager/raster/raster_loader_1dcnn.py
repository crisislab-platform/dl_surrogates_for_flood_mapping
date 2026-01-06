import logging
import os
import numpy as np
import pandas as pd
import rasterio as rio
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from modules.lib.constants import DATA_DIR, SIMULATION_DATA_DIR, DEM_FILE
from modules.datamanager.datamanager import DataManager
from modules.utils.run_util import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataLoader")

elevation_file_path = DEM_FILE
bc_data_dir = f"{DATA_DIR}"
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
        #remove duplicates
        self.all_event_ids = np.unique(self.all_event_ids)
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
        self.device = check_device()
        # Calculate features: 3 upstream flows × lag steps + original 3 upstream flows + time column
        self.features = (lag * 3) + 3 + 1
        self.outputs = self.find_output_size()
        self.pinn = pinn
        self.event_input_map = {}
        self.event_batch_map = {}
        self.inundation_data_cache = {}
        self.idxs_per_event = lambda x: np.ceil((self.get_no_time_steps(x) - self.timestep//2) / self.timestep).astype(int)
        
        try:
            self.idx_prep()
            self.inflow_data_prep(self.all_event_ids)
            self.preload_inundation_data(self.all_event_ids)
            self.prep_test_ouput_data()
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
        return len(inundation_files) - 8
    def preload_inundation_data(self, event_ids):
        """Preload and cache all inundation data for specified event IDs"""
        logger.info(f"Preloading inundation data for events: {event_ids}")
        for event_id in event_ids:
            event_inundation_files = glob.glob(f"{lisflood_simulation_dir}/Run{event_id}-*.wd")
            event_inundation_files.sort()
            event_inundation_files = event_inundation_files[8:]  # Exclude the first 8 files
            # event_inundation_files = event_inundation_files[self.lag:]  # Exclude first lag files
            
            # Initialize cache for this event
            self.inundation_data_cache[event_id] = []
            
            # Process and cache all files for this event
            for i, file in enumerate(event_inundation_files):
                inundation_data = self.process_inundation_file(file)
                if inundation_data is not None:
                    # Convert to tensor and move to GPU if available
                    tensor_data = torch.from_numpy(inundation_data).float()
                    if torch.cuda.is_available() and self.device != 'cpu':
                        tensor_data = tensor_data.cuda()
                        # Fix: Create the zero tensor on the same device
                        zero_tensor = torch.tensor(0.0, device=self.device)
                        tensor_data = torch.where(tensor_data < 0.3, zero_tensor, tensor_data)
                    self.inundation_data_cache[event_id].append(tensor_data)
            
            # Debug info to help diagnose device issues
            if len(self.inundation_data_cache[event_id]) > 0:
                first_tensor = self.inundation_data_cache[event_id][0]
                logger.info(f"Event {event_id} tensor device: {first_tensor.device}")
            
            logger.info(f"Preloaded {len(self.inundation_data_cache[event_id])} timesteps for event {event_id} " +
                        f"{'on GPU' if torch.cuda.is_available() and self.device != 'cpu' else 'on CPU'}")
    
    def inflow_data_prep(self, event_ids):
        self.event_input_map = {}
        self.event_data_map_unscaled = {}
        self.time_data_map = {}  # Store time data separately
        
        for idx in range(len(event_ids)):
            event_id = event_ids[idx]
            inflow_file = os.path.join(DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            # inflow_data = inflow_data[8:]  # Skip the first 8 rows
            
            # Create lagged features for each upstream input
            for i in range(1, self.lag + 1):
                inflow_data[f'Upstream1-{i}'] = inflow_data['Upstream1'].shift(i)
                inflow_data[f'Upstream2-{i}'] = inflow_data['Upstream2'].shift(i)
                inflow_data[f'Upstream3-{i}'] = inflow_data['Upstream3'].shift(i)
                 
            inflow_data = inflow_data.dropna()
            
            # Extract the Time column before processing the flow data
            time_data = inflow_data["Time"].astype(np.float32).values / 3600  # Convert to hours
            time_data = time_data.reshape(-1, 1)  # Reshape for proper scaling later
            
            # Drop Time column from flow data for separate processing
            
            # But keep the data in time_data
            flow_data = inflow_data.drop(columns=['Time'])
            
            # Process flow data as before
            input_arr = np.array(flow_data.values.astype(np.float32))
            # Get first 3 columns for unscaled data
            self.event_data_map_unscaled[event_id] = input_arr[:, :3]
            self.event_input_map[event_id] = input_arr
            
            # Store the corresponding time data
            self.time_data_map[event_id] = time_data[:len(input_arr)]  # Ensure same length
               
        # Fit the flow scaler on all training data
        all_train_data = np.vstack([self.event_input_map[id] for id in self.train_event_ids])
        flow_scaler = MinMaxScaler(feature_range=(0, 1))
        flow_scaler.fit(all_train_data)
        
        # Fit a separate time scaler on all training time data
        all_train_time_data = np.vstack([self.time_data_map[id] for id in self.train_event_ids])
        time_scaler = MinMaxScaler(feature_range=(0, 1))
        time_scaler.fit(all_train_time_data)

        for event_id in self.all_event_ids:
            # Scale flow data
            scaled_flow = flow_scaler.transform(self.event_input_map[event_id])
            
            # Scale time data 
            scaled_time = time_scaler.transform(self.time_data_map[event_id])
            
            # Combine time and flow data
            # Add scaled time as the first column
            self.event_input_map[event_id] = np.hstack((scaled_time, scaled_flow))
            
            if event_id in self.test_event_ids:
                input_arr = self.event_input_map[event_id]
                # input_arr = input_arr[self.test_start_index:self.test_end_index + 1]
                input_arr = input_arr.reshape(input_arr.shape[0], 1, input_arr.shape[1])
                self.test_input  = torch.from_numpy(input_arr).float()
                if torch.cuda.is_available() and self.device != 'cpu':
                    self.test_input = self.test_input.cuda()
                
        # Update features count to include Time column
        logger.info(f"Updated feature count: {self.features} (including Time column)")
                
    def process_inundation_file(self, file_path):
        try:
            with rio.open(file_path) as src:
                data = src.read(1)
                return data.flatten()
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            return None
    
    def index_expander(self, event_ids):
        all_indices = []
        if len(event_ids) == 0:
            return np.array([])
        
        # First, collect all valid indices from each event
        for event_id in event_ids:
            start_idx = self.event_start_id_map[event_id]
            num_idx = self.idxs_per_event(event_id)
            event_indices = start_idx + np.arange(num_idx)
            all_indices.extend(event_indices)
            
        # Convert to numpy array and shuffle to mix events
        all_indices = np.array(all_indices)
        rng = np.random.default_rng(42)  # Use same seed for consistency
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
            return np.array([])  # Return empty array as fallback
        
    def idx_prep(self, shuffle=True, random_seed=42):
        logger.info(f"Preparing index for batch size: {self.batch_size}")
        self.event_start_id_map = {}
        
        # Calculate start indices for each event
        for event_id in self.all_event_ids:
            if event_id == 1:  # First event
                start_idx = 0
            else:
                prev_event_id = event_id - 1
                start_idx = self.event_start_id_map[prev_event_id] + self.get_no_time_steps(prev_event_id)
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
            
        
    def shuffle_training_data(self, epoch):
        """Shuffle the training batches at the start of each epoch"""
        rng = np.random.default_rng(42 + epoch)  # Different seed per epoch for reproducibility
        rng.shuffle(self.train_idx)
        logger.info(f"Shuffled training data batches for epoch {epoch}")
            
    def find_event_id(self, idx):
        for event_id, start_idx in self.event_start_id_map.items():
            end_idx = start_idx + self.get_no_time_steps(event_id) - 1
            if start_idx <= idx and idx <= end_idx:
                return event_id
        return None
    
    def find_indices(self, idxs):
        # Modified to handle indices potentially from different events
        event_indices_map = {}
        
        # Group indices by their respective events
        for idx in idxs:
            event_id = self.find_event_id(idx)
            if event_id is None:
                logger.error(f"Could not find event ID for index {idx}")
                continue
                
            if event_id not in event_indices_map:
                event_indices_map[event_id] = []
                
            start_idx = self.event_start_id_map[event_id]
            # Convert global index to event-local index
            local_idx = idx - start_idx
            event_indices_map[event_id].append(local_idx)
        
        # Return the event IDs and their corresponding local indices
        return event_indices_map
    
    def get_batch(self, indices, subset="train"):
        if self.pinn: 
            return self.get_batch_pinn(indices)
        event_indices_map = self.find_indices(indices)
        if not event_indices_map:
            logger.error(f"Could not find event IDs or local indices for indices {indices}")
            return None, None
        
        input_data = []  # Store paired input/output tensors
        output_data  = []
        
        # Process each event's data separately
        for event_id, local_idxs in event_indices_map.items():
            local_idxs = np.array(local_idxs)
            event_data = self.event_input_map[event_id]
            
            # Filter out invalid indices
            valid_mask = local_idxs < len(event_data)
            valid_idxs = local_idxs[valid_mask]
            
            if len(valid_idxs) == 0:
                continue
                
            # Get input data for this event
            event_batch_data = np.array([event_data[idx] for idx in valid_idxs])
            event_batch_data = event_batch_data.reshape(event_batch_data.shape[0], 1, event_batch_data.shape[1])
            input_tensor = torch.from_numpy(event_batch_data).float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
  
            # Get output data for this event
            output_idxs = valid_idxs
            output_tensors = self.inundation_data_cache[event_id]
            
            valid_output_mask = (output_idxs >= 0) & (output_idxs < len(output_tensors))
            valid_output_idxs = output_idxs[valid_output_mask]

            output_tensor = torch.stack([output_tensors[i] for i in valid_output_idxs]) if len(valid_output_idxs) > 0 else torch.tensor([], device=self.device)
            input_data.append(input_tensor)
            output_data.append(output_tensor)
            
        input_batch = torch.cat(input_data, dim=0)
        output_batch = torch.cat(output_data, dim=0)
        
        # Shuffle the input and output batches together
        if input_batch.shape[0] > 1:  # Only shuffle if we have more than one sample
            indices = torch.randperm(input_batch.shape[0])
            input_batch = input_batch[indices]
            output_batch = output_batch[indices]
            logger.debug(f"Shuffled batch with {len(indices)} samples")
        
        return input_batch, output_batch

    def get_batch_pinn(self, indices):
        event_indices_map = self.find_indices(indices)
        input_data = []  # Store paired input/output tensors
        output_data = [] #yt
        yt_plus1_data = []
        yt_minus1_data = []
        bc_data = []
        bc_plus1_data = []
        
        # Process each event's data separately
        for event_id, local_idxs in event_indices_map.items():
            local_idxs = np.array(local_idxs)
            event_data = self.event_input_map[event_id]
            
            # Filter out invalid indices
            valid_mask = local_idxs < len(event_data)
            valid_idxs = local_idxs[valid_mask]
            
            if len(valid_idxs) == 0:
                continue
                
            # Get input data for this event
            event_batch_data = np.array([event_data[idx] for idx in valid_idxs])
            event_batch_data = event_batch_data.reshape(event_batch_data.shape[0], 1, event_batch_data.shape[1])
            input_tensor = torch.from_numpy(event_batch_data).float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
  
            # Get output data for this event
            inundation_output = self.inundation_data_cache[event_id]
            
            output_idxs = valid_idxs
            valid_output_mask = (output_idxs >= 0) & (output_idxs < len(inundation_output))
            valid_output_idxs = valid_idxs[valid_output_mask]

            # Handle next timestep data for PINN
            next_idxs = np.array([idx + 1 for idx in valid_output_idxs])
            prev_idxs = np.array([idx - 1 for idx in valid_output_idxs])
            
            valid_next_mask = (next_idxs >= 0) & (next_idxs < len(inundation_output))
            valid_prev_mask = (prev_idxs >= 0) & (prev_idxs < len(inundation_output))
            
            valid_output_idxs = valid_output_idxs[valid_next_mask & valid_prev_mask]
            next_idxs = next_idxs[valid_next_mask & valid_prev_mask]
            prev_idxs = prev_idxs[valid_next_mask & valid_prev_mask]
            
            input_tensor = input_tensor[valid_next_mask & valid_prev_mask]
            output_tensor = torch.stack([inundation_output[i] for i in valid_output_idxs])
            yt_minus1_tensor = torch.stack([inundation_output[i] for i in prev_idxs])
            yt_plus1_tensor = torch.stack([inundation_output[i] for i in next_idxs])
            
            # Get boundary conditions
            bct_tensor = torch.from_numpy(self.get_boundary_conditions(event_id, valid_output_idxs)).to(self.device)
            bct_plus1_tensor = torch.from_numpy(self.get_boundary_conditions(event_id, next_idxs) if len(next_idxs) > 0 else np.array([])).to(self.device)
            
            # Append to our collection
            input_data.append(input_tensor)
            output_data.append(output_tensor)
            yt_plus1_data.append(yt_plus1_tensor)
            bc_data.append(bct_tensor)
            bc_plus1_data.append(bct_plus1_tensor)
            yt_minus1_data.append(yt_minus1_tensor)
            
        # Combine all event data
        input_batch = torch.cat(input_data, dim=0) if input_data else torch.tensor([], device=self.device)
        output_batch = torch.cat(output_data, dim=0) if output_data else torch.tensor([], device=self.device)
        yt_minus1_batch = torch.cat(yt_minus1_data, dim=0) if yt_minus1_data else torch.tensor([], device=self.device)
        yt_plus1_batch = torch.cat(yt_plus1_data, dim=0) if yt_plus1_data else torch.tensor([], device=self.device)
        bct_batch = torch.cat(bc_data, dim=0) if bc_data else torch.tensor([], device=self.device)
        bct_plus1_batch = torch.cat(bc_plus1_data, dim=0) if bc_plus1_data else torch.tensor([], device=self.device)
        
        # Shuffle data
        if input_batch.shape[0] > 1:
            indices = torch.randperm(input_batch.shape[0])
            input_batch = input_batch[indices]
            output_batch = output_batch[indices]
            yt_minus1_batch = yt_minus1_batch[indices]
            yt_plus1_batch = yt_plus1_batch[indices]
            bct_batch = bct_batch[indices]
            bct_plus1_batch = bct_plus1_batch[indices]
            logger.debug(f"Shuffled PINN batch with {len(indices)} samples")
        
        return input_batch, output_batch, yt_minus1_batch, yt_plus1_batch, bct_batch, bct_plus1_batch
    
    def get_boundary_conditions(self, event_id, indices):
        bc_data = np.zeros((len(indices), 3))
        event_data = self.event_data_map_unscaled[event_id]
        for i, idx in enumerate(indices):
            if idx < len(event_data):
                bc_data[i, 0] = event_data[idx, 0]
                bc_data[i, 1] = event_data[idx, 1]
                bc_data[i, 2] = event_data[idx, 2]
            else:
                logger.warning(f"Index {idx} out of bounds for event {event_id}")
                exit(1)
        return bc_data

    def prep_test_ouput_data(self):
        test_output = self.inundation_data_cache[self.test_event_ids[0]]
        # test_output = test_output[self.test_start_index:self.test_end_index + 1]
        #crete a tensor from the test output
        test_output = torch.stack(test_output) if len(test_output) > 0 else torch.tensor([], device=self.device)
        if torch.cuda.is_available():
                self.test_output = test_output.cuda()
        logger.info(f"Test output data shape: {len(self.test_output)}")
