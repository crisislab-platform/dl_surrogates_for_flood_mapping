import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray, read_shp_point, coords2rc, gdal_transform
import torch
import logging
import glob
from modules.utils.run_util import check_device
from modules.datamanager.datamanager import DataManager
import rasterio as rio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataManager")

class CNNSequentialDataManager(DataManager):
    def __init__(self, batch_size=32, input_time_len_h=1, rl_group=1, timestep=1, sampling_dist=20, num_of_clusters=100, fold=None, tuning_mode=True):
        super().__init__()
        # Directories
        self.simulation_data_dir = SIMULATION_DATA_DIR
        self.dem_file = os.path.join(self.simulation_data_dir, "Carlisle_5m.asc")
        
        # Prepare train, test, and validation event ids
        self.all_event_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if tuning_mode:
            validation_event = fold + 1
            self.train_event_ids = [event for event in self.all_event_ids if event != validation_event and event != 1]
            self.test_event_ids = [1]
            self.val_event_ids = [validation_event]
          
        else:   
            self.train_event_ids = [2, 3, 4, 5, 6, 7, 8, 9]
            self.test_event_ids = [1]
            self.val_event_ids = [1]
            
        # Sequence variables
        self.batch_size = batch_size
        self.tuning_mode = tuning_mode
        self.timestep = timestep # Time step multipler against the original data (15 min * multipler)
        self.event_seq_data = {}
        self.input_seq_length = int(input_time_len_h * 4 / timestep) 
        
        # We're not shifting the output index now - the output corresponds to the last element in the input sequence
        # This means input sequence (t-23, t-22, ..., t) is used to predict the output at time t
        self.input_seq_start = 0  # No offset needed since we're aligning directly
        
        self.time_slicing = lambda x: x[self.timestep//2::self.timestep] #start:stop:step
        
        # RL variables
        self.rl_group_size = None
        self.rl_group = rl_group
        self.rl_filter = None
        self.sampling_dist = sampling_dist
        self.num_of_clusters = num_of_clusters
        self.rl_cluster_file = self.find_cluster_file(self.rl_group, self.sampling_dist, self.num_of_clusters)

        # Prepare the filter mask for representative locations considered by the RL group in the clusters
        self.prepare_rl_filter(self.rl_group)
        self.idxs_per_event = lambda x: np.ceil((self.get_no_time_steps(x) - self.timestep//2) / self.timestep).astype(int)
        self.event_batch_map = {}
        
        
        self.prepare_batch_idxs()
        self.input_tensor_prep()
        # Preload all inundation data to GPU memory
        self.inundation_data_cache = {}
        logger.info("Preloading inundation data to GPU...")
        for event_id in self.all_event_ids:
            self.preload_inundation_data(event_id)     
        self.debug_data_alignment()
        
        if not self.tuning_mode:
            # if event_id in self.test_event_ids:
            seq_start_index = self.test_start_index - self.input_seq_length + 1
            seq_end_index = self.test_end_index - self.input_seq_length + 1
            test_input = self.event_input_map[self.test_event_ids[0]]
            test_input = test_input[seq_start_index:seq_end_index + 1, :, :]
            self.test_input = torch.from_numpy(test_input).float()
            if torch.cuda.is_available():
                self.test_input = self.test_input.cuda()
            self.prep_test_ouput_data()
            
    def preload_inundation_data(self, event_id):
        """Preload all inundation data for an event into GPU memory"""
        event_inundation_files = glob.glob(f"{self.simulation_data_dir}/Run{event_id}-*.wd")
        event_inundation_files.sort()
        event_inundation_files = event_inundation_files[8:]  # Skip first 8 filess
        
        if not event_inundation_files:
            logger.warning(f"No inundation files found for event {event_id}")
            return
            
        logger.info(f"Loading {len(event_inundation_files)} inundation files for event {event_id}")
        
        # Load all files for this event
        event_inundation_data = []
        for file in event_inundation_files:
            inundation_data = gdal_asarray(file)
            inundation_data = self.rl_filter(inundation_data)
            event_inundation_data.append(inundation_data)
            
        event_inundation_data = np.array(event_inundation_data)

        # Set negligible inundation to 0
        event_inundation_data[event_inundation_data < 0.3] = 0
        
        # Convert to tensor and move to GPU
        tensor_data = torch.from_numpy(event_inundation_data).float()
        
        if torch.cuda.is_available():
            tensor_data_moved = tensor_data.cuda()
            logger.info(f"Loaded inundation data to GPU for event {event_id}, shape: {tensor_data.shape}")
        else:
            logger.info(f"CUDA not available, loaded inundation data to CPU for event {event_id}")
            
        # Store in cache
        self.inundation_data_cache[event_id] = tensor_data_moved
        
    def get_batch(self, indices):
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
            input_tensor = torch.from_numpy(event_batch_data).float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
  
            # Get output data for this event
            output_idxs = valid_idxs + self.input_seq_length - 1
            output_tensors = self.inundation_data_cache[event_id]
                
            valid_output_mask = (output_idxs >= 0) & (output_idxs < len(output_tensors))
            valid_output_idxs = output_idxs[valid_output_mask]

            output_tensor = output_tensors[valid_output_idxs]
            input_data.append(input_tensor)
            output_data.append(output_tensor)
            
        input_batch = torch.cat(input_data, dim=0)
        output_batch = torch.cat(output_data, dim=0)
        
        # Shuffle the input and output batches together
        if input_batch.shape[0] > 1:  # Only shuffle if we have more than one sample
            # Create a random permutation
            indices = torch.randperm(input_batch.shape[0])
            
            # Apply the same permutation to both input and output
            input_batch = input_batch[indices]
            output_batch = output_batch[indices]
            
            logger.debug(f"Shuffled batch with {len(indices)} samples")
        
        return input_batch, output_batch
        
    def load_inundation_data(self, event_id, time_indices):
        """Fallback method to load inundation data if not in cache"""
        logger.warning(f"Using fallback inundation data loading for event {event_id}")
        event_inundation_files = glob.glob(f"{self.simulation_data_dir}/Run{event_id}-*.wd")
        event_inundation_files.sort()
        event_inundation_files = event_inundation_files[8:]  # Skip first 8 files
        
        if event_id == self.test_event_ids[0]:
            logger.info(f"Loading inundation data for test event {event_id}")
            event_inundation_files = event_inundation_files[self.test_start_index:self.test_end_index + 1]
        
        # File loading still needs to happen on CPU
        valid_indices = []
        valid_files = []
        
        # Check for valid indices
        for i in time_indices:
            if 0 <= i < len(event_inundation_files):
                valid_indices.append(i)
                valid_files.append(event_inundation_files[i])
            else:
                logger.warning(f"Invalid index {i} for event {event_id}. Max index: {len(event_inundation_files)-1}")
        
        if not valid_files:
            logger.error(f"No valid inundation files found for event {event_id} with indices {time_indices}")
            # Return empty tensor with correct dimensions
            return torch.zeros((0, self.rl_group_size)).cuda() if torch.cuda.is_available() else torch.zeros((0, self.rl_group_size))
        
        # Load data on CPU first
        event_inundation_data = []
        for file in valid_files:
            inundation_data = gdal_asarray(file)
            inundation_data = self.rl_filter(inundation_data)
            event_inundation_data.append(inundation_data)
            
        event_inundation_data = np.array(event_inundation_data)
        tensor_data = torch.from_numpy(event_inundation_data).float()
        if torch.cuda.is_available():
            tensor_data = tensor_data.cuda()
        
        # Set negligible inundation to 0 - this now happens on GPU
        tensor_data = torch.where(tensor_data < 0.2, torch.tensor(0.0).to(tensor_data.device), tensor_data)
        return tensor_data
        
    def prep_test_ouput_data(self):
        if self.test_event_ids[0] in self.inundation_data_cache:
            test_output = self.inundation_data_cache[self.test_event_ids[0]]
            test_output = test_output[self.test_start_index:self.test_end_index + 1]
            if torch.cuda.is_available():
                self.test_output = test_output.cuda()
        else:
            # Fallback to loading from disk
            logger.warning("Test event not found in cache, loading from disk")
      
        
    def prepare_event_start_idx_map(self):
        event_start_id_map = {}
        for index, event_id in enumerate(self.all_event_ids):
            if index == 0:
                start_idx = 0
            else:
                start_idx = event_start_id_map[self.all_event_ids[index-1]] + self.idxs_per_event(self.all_event_ids[index-1])
            event_start_id_map[event_id] = start_idx
        self.event_start_id_map = event_start_id_map
        
    def prepare_rl_filter(self, rl_group):
        logger.info(f"Preparing RL filter for group {rl_group}")
        
        if rl_group is None:
            self.rl_filter = lambda x: x
            return
        
        self.cluster_rls = self.find_cluster_rls(self.rl_cluster_file, rl_group)
        self.rl_group_size = len(self.cluster_rls)
        
        transform  = gdal_transform(self.dem_file)
        self.dem_map = gdal_asarray(self.dem_file)
        self.coords_to_cluster_rls = [coords2rc(transform, coord)  for coord in self.cluster_rls]
        
        self.cluster_rl_map = np.zeros(self.dem_map.shape)
        for row, col in self.coords_to_cluster_rls:
            self.cluster_rl_map[row, col] = 1
       
        filter_mask  = self.cluster_rl_map == 1
        self.rl_filter = lambda x: x[filter_mask]
        logger.info(f"RL filter prepared.")

    def prepare_batch_idxs(self):
        idx_expander = lambda x: self.index_expander_func(x)
        self.prepare_event_start_idx_map()
        train_idx_batches = idx_expander(self.train_event_ids)
        validation_idx_batches = idx_expander(self.val_event_ids)
        
        random_seed = 341
        rng = np.random.default_rng(random_seed)
        rng.shuffle(train_idx_batches)
        
        self.train_idx = train_idx_batches
        self.validation_idx = validation_idx_batches
        self.train_batches = len(train_idx_batches)
        self.val_batches = len(validation_idx_batches)
        logger.info(f"Created {self.train_batches} training batches and {self.val_batches} validation batches")
        
    def prepare_test_output(self):
        return None
        
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

    def index_expander_func(self, event_ids):
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
            return np.array([])  # Return empty array as fallback
        
    def input_tensor_prep(self):
        self.event_input_map = {}
        for event_id in self.all_event_ids:
            inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            inflow_data = inflow_data.iloc[8:,:] # Skip the first 8 rows
            inflow_data = inflow_data.values
            inflow_data = inflow_data[:, 1:] # Skip the first column
            inflow_data = inflow_data.astype(float)
            input_arr = np.array(inflow_data)
            input_arr = self.time_slicing(input_arr)
            
            # We need to make sure we have enough data for both input and output
            # If input is from 0 to 23, output is at 24, so we need at least 25 timesteps
            
            max_idx = len(input_arr) - self.input_seq_length
            n_samples = max_idx
            n_features = input_arr.shape[1]
            
            event_sequences = np.zeros((n_samples, self.input_seq_length, n_features))
            # Fill the array with sliding window sequences
            for i in range(self.input_seq_length):
                event_sequences[:, i, :] = input_arr[i:i+n_samples, :]
            
            # Store the input sequences in the event_input_map
            self.event_input_map[event_id] = event_sequences
        self.event_input_map = self.normalise_data(self.event_input_map)
     
    def normalise_data(self, input_map):
        """
        Normalize input data using StandardScaler for each feature
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Create a copy to avoid modifying original data
        normalized_map = {}
        
        # Collect all data to fit the scaler
        all_data = []
        for event_id, event_data in input_map.items():
            # Reshape to 2D for scaling: (samples * seq_length, features)
            samples, seq_length, features = event_data.shape
            reshaped_data = event_data.reshape(-1, features)
            all_data.append(reshaped_data)
        
        # Combine all data and fit scaler
        combined_data = np.vstack(all_data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(combined_data)
        
        # Now normalize each event's data
        for event_id, event_data in input_map.items():
            samples, seq_length, features = event_data.shape
            reshaped_data = event_data.reshape(-1, features)
            
            # Apply scaling
            normalized_data = scaler.transform(reshaped_data)
            
            # Reshape back to original 3D shape: (samples, seq_length, features)
            normalized_data = normalized_data.reshape(samples, seq_length, features)
            
            # Store in the new map
            normalized_map[event_id] = normalized_data
            
        # Store scaler for potential inverse transform later
        self.scaler = scaler
        logger.info("Data normalization completed using MinMaxScaler")
        
        return normalized_map

    def get_no_time_steps(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_data_dir}/Run{event_id}-*.wd")
        # Adjust calculation - we need enough timesteps for full sequences plus one more for output
        return len(inundation_files) - 8 - self.input_seq_length
    
    def find_cluster_file(self, rl_group, sampling_dist, num_of_clusters):
        if rl_group is None:
            return None, None
        cluster_file = f"{OUTPUT_DIR}/rls/clusters/clusters_ss_{sampling_dist}_{num_of_clusters}.csv"
        if not os.path.exists(cluster_file):
            logger.error(f"Cluster file {cluster_file} does not exist.")
            raise ValueError(f"Cluster file {cluster_file} does not exist.")
        return cluster_file
    
    def find_cluster_rls(self, cluster_file, rl_group):
        cluster_df = pd.read_csv(cluster_file)
        filtered_df = cluster_df[cluster_df['cluster'] == int(rl_group)]

        coords_list = list(zip(filtered_df['x'], filtered_df['y']))
        logger.info(f"Found {len(coords_list)} representative locations for cluster {rl_group}")
        return coords_list
    
    def process_inundation_file(self, file_path):
        try:
            with rio.open(file_path) as src:
                data = src.read(1)
                return data.flatten()
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            return None
    
    def debug_data_alignment(self):
        """Debug function to check input/output data alignment"""
        event_id = self.train_event_ids[0]
        local_idx = 10  # Choose some reasonable index
        
        # Get input data
        input_data = self.event_input_map[event_id][local_idx]
        
        # Get corresponding output data
        output_idx = local_idx + self.input_seq_length - 1
        if event_id in self.inundation_data_cache and output_idx < len(self.inundation_data_cache[event_id]):
            output_data = self.inundation_data_cache[event_id][output_idx]
            
            # Print stats
            logger.info(f"Input data shape: {input_data.shape}, non-zero: {np.count_nonzero(input_data)}")
            logger.info(f"Output data shape: {output_data.shape}, non-zero: {torch.count_nonzero(output_data).item()}")
            logger.info(f"Output data max: {output_data.max().item()}, min: {output_data.min().item()}")
        else:
            logger.error(f"Cannot access output data for event {event_id}, index {output_idx}")
