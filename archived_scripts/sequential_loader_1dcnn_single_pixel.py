import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray, read_shp_point, coords2rc, gdal_transform
import torch
import logging
from modules.datamanager.datamanager import DataManager, check_inundation_data_cache
from modules.datamanager.raster.raster_loader_usrr import ReconsturctionDataManager
import rasterio as rio
from torch.utils.data import DataLoader
import glob
from modules.utils.run_util import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataManager")

class CNNSequentialDataManager(DataManager):
    def __init__(self, batch_size=32, input_time_len_h=1, rl_group=1, sampling_dist=20, num_of_clusters=100, fold=1, 
                 tuning_mode=True, reconstruction_mode=False, reco_data_manager:ReconsturctionDataManager=None):
        torch.cuda.empty_cache()
        with torch.no_grad():
            super().__init__()
            if rl_group == 1 or rl_group == '0': 
                logger.info("Using all representative locations for single-pixel training")
                
            # Directories
            self.simulation_data_dir = SIMULATION_DATA_DIR
            self.dem_file = os.path.join(self.simulation_data_dir, "Carlisle_5m.asc")
            self.reconstruction_mode = reconstruction_mode
            self.tuning_mode = tuning_mode
            self.device = check_device()
            
            # Prepare train, test, and validation event ids
            if self.tuning_mode:
                self.all_event_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                validation_event = fold + 1
                self.train_event_ids = [event for event in self.all_event_ids if event != validation_event and event != 1]
                self.test_event_ids = [1]
                self.val_event_ids = [validation_event]
            elif self.reconstruction_mode:
                self.reco_data_manager = reco_data_manager
                self.all_event_ids = [1]
                self.train_event_ids = [1]
                self.test_event_ids = [1]
                self.val_event_ids = []
            else:   
                self.all_event_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                self.train_event_ids = [2, 3, 4, 5, 6, 7, 8, 9]
                self.test_event_ids = [1]
                self.val_event_ids = []
                
            # Sequence variables
            self.batch_size = batch_size
            self.tuning_mode = tuning_mode
            self.event_seq_data = {}
            self.input_seq_length = int(input_time_len_h * 4)
            
            # RL variables - now for single pixel training
            self.rl_group_size = 1  # Output is now 1-dimensional (single RL location)
            self.rl_group = rl_group
            self.rl_filter = None
            self.sampling_dist = sampling_dist
            self.num_of_clusters = num_of_clusters
            self.rl_cluster_file = self.find_cluster_file(self.rl_group, self.sampling_dist, self.num_of_clusters)
            self.event_batch_map = {}
            self.inundation_data_cache = {}
            
            # Single pixel training variables
            self.rl_locations = []  # List of (row, col) tuples for each RL
            self.rl_elevations = []  # Elevation at each RL location
            self.num_rl_locations = 0
            
            # Training data will be organized per RL location
            self.train_sequences_by_rl = {}  # {rl_idx: {event_id: sequences}}
            self.val_sequences_by_rl = {}
            self.test_sequences_by_rl = {}
            self.train_outputs_by_rl = {}    # {rl_idx: {event_id: outputs}}
            self.val_outputs_by_rl = {}
            self.test_outputs_by_rl = {}

            # Prepare the filter mask for representative locations
            self.prepare_rl_filter(self.rl_group)
            self.preload_inundation_data()
            self.input_tensor_prep_single_pixel()
            
            if not self.tuning_mode:
                with torch.no_grad():
                    # For testing, we'll use all RL locations
                    self.test_input_by_rl = self.test_sequences_by_rl
                    self.test_output_by_rl = self.test_outputs_by_rl
                
            if self.reconstruction_mode:
                # Handle reconstruction mode for single pixel
                self.setup_reconstruction_mode()
            else:
                self.prepare_batch_idxs_single_pixel()
            
        
    def preload_inundation_data(self):
        torch.cuda.empty_cache()
        with torch.no_grad():
            """Preload all inundation data for an event into GPU memory"""
            for event_id in self.all_event_ids:
                if check_inundation_data_cache(event_id):
                    inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{event_id}_inundation.pt"))
                    inundation_data = inundation_data.float()
                    event_inundation_data = inundation_data[:, self.filter_mask]
                    # if event_id == 1:
                    #     self.visualise_depth_curve(event_inundation_data)
                    self.inundation_data_cache[event_id] = event_inundation_data
                    logger.info(f"inundation cache size:  {self.inundation_data_cache[event_id].shape}")
                    logger.info(f"Loaded inundation data for event {event_id} from cache")
                else:
                    raise ValueError(f"Inundation data for event {event_id} not found in cache. Please run create_inundation_map_tensors() first.")
    
    def prepare_rl_filter(self, rl_group):
        self.rl_filter = None
        self.cluster_rls = self.find_cluster_rls(self.rl_cluster_file, rl_group)
        self.dem_map = gdal_asarray(self.dem_file)
        self.cluster_rl_map = np.zeros(self.dem_map.shape)

        # Mark all representative locations in the map
        for row, col in self.cluster_rls:
            self.cluster_rl_map[row, col] = 1
            self.rl_locations.append((row, col))
            self.rl_elevations.append(self.dem_map[row, col])
       
        self.num_rl_locations = len(self.cluster_rls)
        self.filter_mask = self.cluster_rl_map == 1
        
        # Get the elevation values at the representative locations
        self.dem_vector_rls = np.array(self.rl_elevations)
        
        # Create filter functions
        self.rl_filter = lambda x: x[self.filter_mask]
        self.rl_filter_tensor = lambda x: torch.masked_select(x, torch.tensor(self.filter_mask).cuda())
        
        logger.info(f"RL filter prepared with {self.num_rl_locations} representative locations")
        logger.info(f"Filter mask shape: {self.filter_mask.shape}")
        logger.info(f"Elevation range: {np.min(self.dem_vector_rls):.2f}m - {np.max(self.dem_vector_rls):.2f}m")
    
        # Visualize the representative locations on the DEM
        self.visualize_rls_cluster()
        
    def visualize_rls_cluster(self):
        """Visualize the representative locations on the DEM for single pixel training"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot DEM
            im = ax.imshow(self.dem_map, cmap='terrain', alpha=0.7)
            
            # Plot RL locations
            if len(self.rl_locations) > 0:
                rl_rows, rl_cols = zip(*self.rl_locations)
                ax.scatter(rl_cols, rl_rows, c='red', s=20, marker='o', alpha=0.8, 
                          label=f'RL Locations ({self.num_rl_locations})')
            
            ax.set_title(f'Representative Locations for Single Pixel Training\n'
                        f'Group: {self.rl_group}, Sampling Distance: {self.sampling_dist}m, '
                        f'Clusters: {self.num_of_clusters}')
            ax.legend()
            
            # Add colorbar for elevation
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Elevation (m)')
            
            # Save visualization
            output_dir = os.path.join(OUTPUT_DIR, 'rl_visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, 
                f'rl_locations_single_pixel_group_{self.rl_group}_dist_{self.sampling_dist}_clusters_{self.num_of_clusters}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"RL visualization saved to {output_file}")
            
        except Exception as e:
            logger.warning(f"Could not create RL visualization: {e}")
    

    def prepare_batch_idxs_single_pixel(self):
        """
        Create batches for single pixel training.
        Combine all RL locations and events into one training dataset.
        """
        # Flatten all training sequences from all RL locations into one dataset
        all_train_sequences = []
        all_train_outputs = []
        all_val_sequences = []
        all_val_outputs = []
        
        # Training data
        for rl_idx in range(self.num_rl_locations):
            for event_id in self.train_event_ids:
                if event_id in self.train_sequences_by_rl[rl_idx]:
                    sequences = self.train_sequences_by_rl[rl_idx][event_id]
                    outputs = self.train_outputs_by_rl[rl_idx][event_id]
                    
                    for i in range(len(sequences)):
                        all_train_sequences.append(sequences[i])
                        all_train_outputs.append(outputs[i])
        
        # Validation data
        for rl_idx in range(self.num_rl_locations):
            for event_id in self.val_event_ids:
                if event_id in self.val_sequences_by_rl[rl_idx]:
                    sequences = self.val_sequences_by_rl[rl_idx][event_id]
                    outputs = self.val_outputs_by_rl[rl_idx][event_id]
                    
                    for i in range(len(sequences)):
                        all_val_sequences.append(sequences[i])
                        all_val_outputs.append(outputs[i])
        
        # Convert to tensors
        self.all_train_sequences = torch.stack(all_train_sequences)
        self.all_train_outputs = torch.stack(all_train_outputs)
        
        if all_val_sequences:
            self.all_val_sequences = torch.stack(all_val_sequences)
            self.all_val_outputs = torch.stack(all_val_outputs)
        else:
            self.all_val_sequences = torch.empty(0)
            self.all_val_outputs = torch.empty(0)
        
        # Create batch indices for training
        self.train_idx = self.index_expander_func("train")
        
        # Create batch indices for validation if we have validation data
        if len(all_val_sequences) > 0:
            self.val_idx = self.index_expander_func("val")
        else:
            self.val_idx = []
        
        logger.info(f"Single pixel batch preparation completed:")
        logger.info(f"  Total training samples: {len(all_train_sequences)}")
        logger.info(f"  Total validation samples: {len(all_val_sequences)}")
        logger.info(f"  Training batches: {len(self.train_idx)}")
        logger.info(f"  Validation batches: {len(self.val_idx) if hasattr(self, 'val_idx') else 0}")
        
    def index_expander_func(self, subset, epoch=0, event_id=None):
        """Create batch indices for single pixel training"""
        if subset == "train":
            all_indices = np.arange(len(self.all_train_sequences))
        elif subset == "val":
            all_indices = np.arange(len(self.all_val_sequences))
        else:
            raise ValueError(f"Invalid subset: {subset}")
        
        # Shuffle indices
        rng = np.random.default_rng(42)  # Fixed seed for reproducible results
        rng.shuffle(all_indices)
        
        # Create batches
        num_of_batches = len(all_indices) // self.batch_size
        idx_batch_list = []
        
        for i in range(num_of_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(all_indices))
            if batch_end - batch_start == self.batch_size:  # Only full batches
                batch_indices = all_indices[batch_start:batch_end]
                idx_batch_list.append(batch_indices)
        
        if len(idx_batch_list) > 0:
            idx_batch_list = np.array(idx_batch_list)
            logger.info(f"Created {len(idx_batch_list)} {subset} batches with {len(all_indices)} total samples")
            return idx_batch_list
        else:
            logger.warning(f"No {subset} batches created! Check data size and batch size.")
            return np.array([])
    
    def get_batch(self, batch_item, subset="train"):
        """
        Get a batch for single pixel training
        
        Args:
            batch_item: Either batch indices for single pixel mode, or (event_id, indices) for multi-location mode
            subset: "train" or "val"
        """
        if subset == "train":
            # For single pixel training, batch_item is just indices
            if isinstance(batch_item, (list, np.ndarray)) and len(batch_item) > 0:
                input_sequences = self.all_train_sequences[batch_item]
                output_sequences = self.all_train_outputs[batch_item]
            else:
                # Handle the case where batch_item is (event_id, indices) - fallback to old behavior
                event_id, indices = batch_item
                input_sequences = torch.stack([self.all_train_sequences[idx] for idx in indices])
                output_sequences = torch.stack([self.all_train_outputs[idx] for idx in indices])
                
        elif subset == "val":
            if isinstance(batch_item, (list, np.ndarray)) and len(batch_item) > 0:
                input_sequences = self.all_val_sequences[batch_item]
                output_sequences = self.all_val_outputs[batch_item]
            else:
                # Handle the case where batch_item is (event_id, indices) - fallback to old behavior
                event_id, indices = batch_item
                input_sequences = torch.stack([self.all_val_sequences[idx] for idx in indices])
                output_sequences = torch.stack([self.all_val_outputs[idx] for idx in indices])
        else:
            raise ValueError(f"Invalid subset: {subset}. Choose 'train' or 'val'.")
        
        with torch.no_grad():
            input_tensor = input_sequences.float().cuda()
            output_tensor = output_sequences.float().cuda()
            return input_tensor, output_tensor
    
    def setup_reconstruction_mode(self):
        """Setup for reconstruction mode - adapt for single pixel"""
        if self.reconstruction_mode and self.reco_data_manager:
            # For single pixel reconstruction, we need to handle the data differently
            self.inundation_data_cache[self.test_event_ids[0]] = self.reco_data_manager.preloaded_maps
            logger.info("Single pixel reconstruction mode setup completed")
    
    def get_test_input_output_by_rl(self, rl_idx):
        """Get test input and output for a specific RL location"""
        if rl_idx >= self.num_rl_locations:
            raise ValueError(f"RL index {rl_idx} out of range [0, {self.num_rl_locations})")
        
        test_sequences = []
        test_outputs = []
        
        for event_id in self.test_event_ids:
            if event_id in self.test_sequences_by_rl[rl_idx]:
                sequences = self.test_sequences_by_rl[rl_idx][event_id]
                outputs = self.test_outputs_by_rl[rl_idx][event_id]
                
                for i in range(len(sequences)):
                    test_sequences.append(sequences[i])
                    test_outputs.append(outputs[i])
        
        if test_sequences:
            return torch.stack(test_sequences), torch.stack(test_outputs)
        else:
            return torch.empty(0), torch.empty(0)
    
    def get_all_test_data(self):
        """Get all test data combined from all RL locations"""
        all_test_sequences = []
        all_test_outputs = []
        
        for rl_idx in range(self.num_rl_locations):
            sequences, outputs = self.get_test_input_output_by_rl(rl_idx)
            if len(sequences) > 0:
                for i in range(len(sequences)):
                    all_test_sequences.append(sequences[i])
                    all_test_outputs.append(outputs[i])
        
        if all_test_sequences:
            return torch.stack(all_test_sequences), torch.stack(all_test_outputs)
        else:
            return torch.empty(0), torch.empty(0)

        
    def input_tensor_prep(self):
        with torch.no_grad():
            # First load and normalize raw inflow data for each event
            raw_inflow_data = {}
            all_train_data = []
            
            # 1. Load all raw data
            for event_id in self.all_event_ids:
                inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
                inflow_data = pd.read_csv(inflow_file)
                
                inflow_data = inflow_data
                inflow_data = inflow_data.values
                inflow_data = inflow_data[8:, 1:] 
                input_arr = np.array(inflow_data)
                
                # Add padding to the input array from the left
                raw_inflow_data[event_id] = input_arr
                
                # Only collect training data for fitting the scaler
                if event_id in self.train_event_ids:
                    all_train_data.append(input_arr)
            
            # 2. Normalize the data using MinMaxScaler feature by feature
            from sklearn.preprocessing import MinMaxScaler
            
            # Combine all training data
            all_train_data = np.vstack(all_train_data)
            
            # Create and fit a scaler for each feature independently
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_train_data)
            
            # Store sequences by event to avoid mixing events in batches
            self.train_sequences_by_event = {}  # Dictionary to store sequences by event ID
            self.val_sequences_by_event = {}    # Dictionary to store validation sequences by event ID
            self.test_sequences_by_event = {}   # Dictionary to store test sequences by event ID
            
            # Also maintain combined sequences for backward compatibility
            self.train_sequences = []
            for event_id, raw_data in raw_inflow_data.items():
                normalized_data = self.scaler.transform(raw_data)
                padding_length = self.input_seq_length - 1
                normalized_data = np.r_['0,2', np.zeros((padding_length, normalized_data.shape[1])), normalized_data]
                normalized_data[:padding_length, :] = np.repeat(normalized_data[padding_length:padding_length+1, :], padding_length, axis=0)  # Fill padding with first valid values
                n_samples = len(normalized_data) - self.input_seq_length + 1
            
                
                
                # Create sequences
                input_sequences = [normalized_data[i: i + self.input_seq_length, :] for i in range(n_samples)]
                input_sequences = np.array(input_sequences)
                
                input_sequences = torch.from_numpy(input_sequences).float()

                # Store sequences by event ID to maintain event separation
                if event_id in self.train_event_ids:
                    self.train_sequences_by_event[event_id] = input_sequences
                if event_id in self.test_event_ids:
                    self.test_sequences_by_event[event_id] = input_sequences
                if event_id in self.val_event_ids:
                    self.val_sequences_by_event[event_id] = input_sequences
                    
    def input_tensor_prep_single_pixel(self):
        """
        Prepare input sequences for single pixel training.
        Each RL location gets its own sequences with features: [upstream1, upstream2, upstream3, elevation]
        Output is 1-dimensional (inundation depth at that specific RL location)
        """
        with torch.no_grad():
            # First load and normalize raw inflow data for each event
            raw_inflow_data = {}
            all_train_data = []
            
            # 1. Load all raw data
            for event_id in self.all_event_ids:
                inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
                inflow_data = pd.read_csv(inflow_file)
                
                inflow_data = inflow_data.values
                inflow_data = inflow_data[8:, 1:]  # Skip header and time column
                input_arr = np.array(inflow_data, dtype=np.float32)
                
                raw_inflow_data[event_id] = input_arr
                
                # Only collect training data for fitting the scaler
                if event_id in self.train_event_ids:
                    all_train_data.append(input_arr)
            
            # 2. Normalize the inflow data using MinMaxScaler
            from sklearn.preprocessing import MinMaxScaler
            
            # Combine all training data for scaler fitting
            all_train_data = np.vstack(all_train_data)
            
            # Create scaler for inflow features (upstream1, upstream2, upstream3)
            self.inflow_scaler = MinMaxScaler()
            self.inflow_scaler.fit(all_train_data)
            
            # Normalize elevation values separately
            self.elevation_scaler = MinMaxScaler()
            self.elevation_scaler.fit(self.dem_vector_rls.reshape(-1, 1))
            
            logger.info(f"Creating sequences for {self.num_rl_locations} RL locations")
            
            # 3. Create sequences for each RL location
            for rl_idx in range(self.num_rl_locations):
                elevation_value = self.dem_vector_rls[rl_idx]
                normalized_elevation = self.elevation_scaler.transform([[elevation_value]])[0, 0]
                
                # Initialize storage for this RL location
                self.train_sequences_by_rl[rl_idx] = {}
                self.val_sequences_by_rl[rl_idx] = {}
                self.test_sequences_by_rl[rl_idx] = {}
                self.train_outputs_by_rl[rl_idx] = {}
                self.val_outputs_by_rl[rl_idx] = {}
                self.test_outputs_by_rl[rl_idx] = {}
                
                for event_id, raw_inflow in raw_inflow_data.items():
                    # Normalize inflow data
                    normalized_inflow = self.inflow_scaler.transform(raw_inflow)
                    
                    # Add padding for sequence creation
                    padding_length = self.input_seq_length - 1
                    normalized_inflow = np.r_['0,2', np.zeros((padding_length, normalized_inflow.shape[1])), normalized_inflow]
                    normalized_inflow[:padding_length, :] = np.repeat(
                        normalized_inflow[padding_length:padding_length+1, :], padding_length, axis=0
                    )
                    
                    n_samples = len(normalized_inflow) - self.input_seq_length + 1
                    
                    # Create input sequences with elevation feature
                    input_sequences = []
                    for i in range(n_samples):
                        # Get inflow sequence [seq_length, 3]
                        inflow_seq = normalized_inflow[i: i + self.input_seq_length, :]
                        
                        # Add elevation as 4th feature to each timestep
                        elevation_column = np.full((self.input_seq_length, 1), normalized_elevation)
                        seq_with_elevation = np.hstack([inflow_seq, elevation_column])
                        
                        input_sequences.append(seq_with_elevation)
                    
                    input_sequences = np.array(input_sequences)
                    input_sequences = torch.from_numpy(input_sequences).float()
                    
                    # Get output sequences (inundation depth at this RL location)
                    inundation_at_rl = self.inundation_data_cache[event_id][:, rl_idx]  # Shape: [timesteps]
                    
                    # Create output sequences matching input sequences
                    output_sequences = []
                    for i in range(n_samples):
                        # Output is the depth at the end of the sequence
                        target_timestep = i + self.input_seq_length - 1
                        if target_timestep < len(inundation_at_rl):
                            output_sequences.append(inundation_at_rl[target_timestep].item())
                        else:
                            output_sequences.append(0.0)  # Fallback for edge cases
                    
                    output_sequences = torch.tensor(output_sequences, dtype=torch.float32)
                    
                    # Store sequences by event and subset
                    if event_id in self.train_event_ids:
                        self.train_sequences_by_rl[rl_idx][event_id] = input_sequences
                        self.train_outputs_by_rl[rl_idx][event_id] = output_sequences
                    elif event_id in self.test_event_ids:
                        self.test_sequences_by_rl[rl_idx][event_id] = input_sequences
                        self.test_outputs_by_rl[rl_idx][event_id] = output_sequences
                    elif event_id in self.val_event_ids:
                        self.val_sequences_by_rl[rl_idx][event_id] = input_sequences
                        self.val_outputs_by_rl[rl_idx][event_id] = output_sequences
                
                if rl_idx % 10 == 0:
                    logger.info(f"Processed RL location {rl_idx}/{self.num_rl_locations}")
            
            logger.info(f"Single pixel sequence preparation completed for {self.num_rl_locations} RL locations")
            logger.info(f"Input feature shape: [batch_size, {self.input_seq_length}, 4] (upstream1, upstream2, upstream3, elevation)")
            logger.info(f"Output shape: [batch_size] (inundation depth at RL location)")
        with torch.no_grad():
            # First load and normalize raw inflow data for each event
            raw_inflow_data = {}
            all_train_data = []
            
            # 1. Load all raw data
            for event_id in self.all_event_ids:
                inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
                inflow_data = pd.read_csv(inflow_file)
                
                # Create lagged features by shifting the data
                for col in inflow_data.columns[1:]:
                    for lag in range(1, 9):  # Create 8 lagged features
                        inflow_data[f'{col}_lag{lag}'] = inflow_data[col].shift(lag)
                        
                #drop rows with NaN values created by shifting
                inflow_data = inflow_data.dropna().reset_index(drop=True)
                input_arr = np.array(inflow_data)
                
                # Add padding to the input array from the left
                raw_inflow_data[event_id] = input_arr
                
                # Only collect training data for fitting the scaler
                if event_id in self.train_event_ids:
                    all_train_data.append(input_arr)
            
            # 2. Normalize the data using MinMaxScaler feature by feature
            from sklearn.preprocessing import MinMaxScaler
            
            # Combine all training data
            all_train_data = np.vstack(all_train_data)
            
            # Create and fit a scaler for each feature independently
            self.flow_scaler = MinMaxScaler()
            self.time_scaler = MinMaxScaler()
            self.flow_scaler.fit(all_train_data[:, 1:])  # Exclude time column from scaling
            self.time_scaler.fit(all_train_data[:, 0].reshape(-1, 1))  # Only scale time column
            
            # Store sequences by event to avoid mixing events in batches
            self.train_sequences_by_event = {}  # Dictionary to store sequences by event ID
            self.val_sequences_by_event = {}    # Dictionary to store validation sequences by event ID
            self.test_sequences_by_event = {}   # Dictionary to store test sequences by event ID
            
            for event_id, raw_data in raw_inflow_data.items():
                normalized_data = self.flow_scaler.transform(raw_data[:, 1:])  # Scale flow features
                normalized_time = self.time_scaler.transform(raw_data[:, 0].reshape(-1, 1))  # Scale time feature
                normalized_data = np.hstack((normalized_time, normalized_data))  #Combine scaled time and flow features
                
                if event_id in self.train_event_ids:
                    self.train_sequences_by_event[event_id] = normalized_data
                    
                if event_id in self.test_event_ids:
                    self.test_sequences_by_event[event_id] = normalized_data
                if event_id in self.val_event_ids:
                    self.val_sequences_by_event[event_id] = normalized_data 
                    
               
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
        row_col_list = list(zip(filtered_df['row'], filtered_df['col']))
        logger.info(f"Found {len(row_col_list)} representative locations for cluster {rl_group}")
        return row_col_list
            
    def visualise_depth_curve(self, inundation_data):
        """Visualize the depth curves for all RL points"""
        import matplotlib.pyplot as plt
        import os
        import matplotlib.cm as cm
        
        # Create visualization directory if it doesn't exist
        vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get the number of RL points
        num_points = inundation_data.shape[1]
        logger.info(f"Visualizing depth curves for all {num_points} RL points")
        
        # Create a time axis (assuming each time step is 15 minutes)
        time_steps = np.arange(inundation_data.shape[0]) * 15 / 60  # Convert to hours
        
        # VISUALIZATION 1: Individual plots for each point
        for rl_point_index in range(num_points):
            if rl_point_index % 10 == 0:
                logger.info(f"Generating plot for RL point {rl_point_index}/{num_points}")
                
            # Extract the depth values for this RL point across all time steps
            depth_values = inundation_data[:, rl_point_index].cpu().numpy()
            
            # Plot the depth curve
            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, depth_values, marker='o', markersize=3)
            plt.title(f'Depth Curve at RL Point Index {rl_point_index}')
            plt.xlabel('Time (hours)')
            plt.ylabel('Depth (m)')
            plt.grid(True)
            
            # Save the plot
            filename = os.path.join(vis_dir, f'depth_curve_rl_point_{rl_point_index}.png')
            plt.savefig(filename, dpi=300)
            plt.close()
        
        # VISUALIZATION 2: All curves on one plot with different colors
        plt.figure(figsize=(16, 10))
        colors = cm.rainbow(np.linspace(0, 1, num_points))
        
        max_depths = []
        # Plot each depth curve with a different color
        for rl_point_index in range(num_points):
            depth_values = inundation_data[:, rl_point_index].cpu().numpy()
            max_depth = np.max(depth_values)
            max_depths.append(max_depth)
            plt.plot(time_steps, depth_values, color=colors[rl_point_index], linewidth=1.5, alpha=0.7)
        
        plt.title(f'Depth Curves for All {num_points} RL Points')
        plt.xlabel('Time (hours)')
        plt.ylabel('Depth (m)')
        plt.grid(True)
        
        # Add a vertical line at peak time (assuming peak is when most points reach max depth)
        peak_hour_index = np.argmax(np.mean(inundation_data.cpu().numpy(), axis=1))
        peak_hour = time_steps[peak_hour_index]
        plt.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.8, 
                   label=f'Peak Time: {peak_hour:.2f} hours')
        
        plt.legend()
        
        # Save the combined plot
        filename = os.path.join(vis_dir, f'depth_curves_all_points.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # VISUALIZATION 3: Heatmap visualization of all points over time
        plt.figure(figsize=(16, 10))
        depth_matrix = inundation_data.cpu().numpy()
        
        # Sort points by maximum depth for better visualization
        sorted_indices = np.argsort(max_depths)[::-1]  # Descending order
        depth_matrix_sorted = depth_matrix[:, sorted_indices]
        
        # Plot heatmap
        im = plt.imshow(depth_matrix_sorted.T, aspect='auto', cmap='viridis', 
                       extent=[0, time_steps[-1], 0, num_points])
        
        plt.colorbar(im, label='Water Depth (m)')
        plt.title(f'Water Depth Heatmap for All {num_points} RL Points')
        plt.xlabel('Time (hours)')
        plt.ylabel('RL Point Index (sorted by maximum depth)')
        
        # Mark the peak time
        plt.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.8)
        
        # Save the heatmap
        filename = os.path.join(vis_dir, f'depth_heatmap_all_points.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualizations for all {num_points} RL points")
    
    def get_feature_stats(self):
        stats = {
            'num_rl_locations': self.num_rl_locations,
            'input_sequence_length': self.input_seq_length,
            'num_features': 4,  # upstream1, upstream2, upstream3, elevation
            'feature_names': ['upstream1', 'upstream2', 'upstream3', 'elevation'],
            'elevation_range': {
                'min': float(np.min(self.dem_vector_rls)),
                'max': float(np.max(self.dem_vector_rls)),
                'mean': float(np.mean(self.dem_vector_rls)),
                'std': float(np.std(self.dem_vector_rls))
            },
            'total_train_samples': len(self.all_train_sequences) if hasattr(self, 'all_train_sequences') else 0,
            'total_val_samples': len(self.all_val_sequences) if hasattr(self, 'all_val_sequences') else 0,
            'output_dimension': 1,
            'output_description': 'Inundation depth at specific RL location'
        }
        return stats