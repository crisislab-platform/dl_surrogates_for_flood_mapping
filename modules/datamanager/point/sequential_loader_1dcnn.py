import os
import pandas as pd
import numpy as np
from modules.lib.constants import DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR, DEM_FILE
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray
import torch
import logging
from modules.datamanager.datamanager import DataManager, check_inundation_data_cache
from modules.datamanager.raster.raster_loader_usrr import ReconsturctionDataManager
from modules.utils.run_util import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataManager")

class CNNSequentialDataManager(DataManager):
    def __init__(self, batch_size=32, input_time_len_h=1, rl_group=1,sampling_dist=20, num_of_clusters=100, fold=1, 
                 tuning_mode=True, reconstruction_mode=False, reco_data_manager:ReconsturctionDataManager=None, stride=1):
        torch.cuda.empty_cache()
        with torch.no_grad():
            super().__init__()
            if rl_group == 1 or rl_group == '0': 
                logger.info("Using all representative locations (RL group 1)")
                
            # Directories
            self.simulation_data_dir = SIMULATION_DATA_DIR
            self.dem_file = DEM_FILE
            self.reconstruction_mode = reconstruction_mode
            self.tuning_mode = tuning_mode
            self.device = check_device()
            
            #Prepare train, test, and validation event ids
            if self.tuning_mode:
                self.all_event_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                validation_event = fold + 1
                self.train_event_ids = [event for event in self.all_event_ids if event != validation_event and event != 1]
                self.test_event_ids = [1]
                self.val_event_ids = [validation_event]
            
            
            elif self.reconstruction_mode:
                self.reco_data_manager = reco_data_manager
                self.all_event_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                self.train_event_ids = [2,3,4,5,6,7,8,9]
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
            self.stride = stride
            
            # RL variables
            self.rl_group_size = None
            self.rl_group = rl_group
            self.rl_filter = None
            self.sampling_dist = sampling_dist
            self.num_of_clusters = num_of_clusters
            self.rl_cluster_file = self.find_cluster_file(self.rl_group, self.sampling_dist, self.num_of_clusters)
            self.event_batch_map = {}
            self.inundation_data_cache = {}

            # Prepare the filter mask for representative locations considered by the RL group in the clusters
            self.prepare_rl_filter(self.rl_group)
            self.preload_inundation_data()
            self.input_tensor_prep()
            
            if not self.tuning_mode:
                with torch.no_grad():
                    self.test_input = self.test_sequences_by_event[self.test_event_ids[0]]
                    self.test_output = self.inundation_data_cache[self.test_event_ids[0]]
                    
                    #Move to GPU
                    self.test_input = self.test_input.to(self.device)
                    self.test_output = self.test_output.to(self.device)
                
            if self.reconstruction_mode:
                self.inundation_data_cache[self.test_event_ids[0]] = self.reco_data_manager.preloaded_maps
            else:
                self.prepare_batch_idxs()
            
        
    def preload_inundation_data(self):
        torch.cuda.empty_cache()
        with torch.no_grad():
            """Preload all inundation data for an event into GPU memory"""
            for event_id in self.all_event_ids:
                if check_inundation_data_cache(event_id):
                    inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{event_id}_inundation.pt"))
                    inundation_data = inundation_data.float()
                    # No need to skip first 8 maps here as they were skipped during preprocessing
                    event_inundation_data = inundation_data[:, self.filter_mask]
                    event_inundation_data = event_inundation_data.to(self.device)
                    # if event_id == 1:
                    #     self.visualise_depth_curve(event_inundation_data)
                    self.inundation_data_cache[event_id] = event_inundation_data
                    logger.info(f"inundation cache size:  {self.inundation_data_cache[event_id].shape}")
                    logger.info(f"Loaded inundation data for event {event_id} from cache")
                else:
                    raise ValueError(f"Inundation data for event {event_id} not found in cache. Please run create_inundation_map_tensors() first.")
        
    def get_batch(self, batch_item, subset="train"):
        event_id, indices = batch_item
        if subset == "train":
                if event_id not in self.train_sequences_by_event:
                    raise ValueError(f"No training sequences for event {event_id}")
                input_sequences = []
                output_sequences = []
                for idx in indices:
                    input_sequences.append(self.train_sequences_by_event[event_id][idx])
                    output_sequences.append(self.train_output_sequences_by_event[event_id][idx])
        elif subset == "val":
            if event_id not in self.val_sequences_by_event:
                raise ValueError(f"No validation sequences for event {event_id}")
            input_sequences = [self.val_sequences_by_event[event_id][idx] for idx in indices]
            output_sequences = [self.inundation_data_cache[event_id][idx] for idx in indices]
        else:
            raise ValueError(f"Invalid subset: {subset}. Choose 'train' or 'val'.")
            
        with torch.no_grad():    
            # Shuffle sequences within the batch
            input_tensor = torch.stack(input_sequences, dim=0).float().to(self.device)
            output_tensor = torch.stack(output_sequences, dim=0).float().to(self.device)
            
            indices = np.arange(len(input_sequences))
            np.random.shuffle(indices)
            input_tensor = input_tensor[indices]
            output_tensor = output_tensor[indices]
            return input_tensor, output_tensor
    
    def shuffle_training_data(self, epoch):
        # Shuffle the training batches at the start of each epoch
        import random
        rng = np.random.default_rng(42 + epoch)  # Different seed per epoch for reproducibility
        rng.shuffle(self.train_idx)
        for event_id in self.train_idx_by_event:
            rng.shuffle(self.train_idx_by_event[event_id])
        logger.info("Shuffled training data batches")
    
    def prepare_rl_filter(self, rl_group):
        self.rl_filter = None
        self.cluster_rls = self.find_cluster_rls(self.rl_cluster_file, rl_group)
        self.dem_map = gdal_asarray(self.dem_file)
        self.cluster_rl_map = np.zeros(self.dem_map.shape)

        # Mark all representative locations in the map (not just the 6th one)
        for row, col in self.cluster_rls:
            self.cluster_rl_map[row, col] = 1
       
        self.rl_group_size = int(self.cluster_rl_map.sum())
        self.filter_mask = self.cluster_rl_map == 1
        
        #Get the elevation values at the representative locations
        self.dem_vector_rls = self.dem_map[self.filter_mask]
        
        # Create filter functions
        self.rl_filter = lambda x: x[self.filter_mask]
        self.rl_filter_tensor = lambda x: torch.masked_select(x, torch.tensor(self.filter_mask).cuda())
        
        logger.info(f"RL filter prepared with {self.rl_group_size} representative locations")
        logger.info(f"Filter mask shape: {self.filter_mask.shape}")
        logger.info(f"Number of True values in filter mask: {np.sum(self.filter_mask)}")
    
        # Visualize the representative locations on the DEM
        # self.visualize_rls_cluster()
        
    def visualize_rls_cluster(self):
        """Visualize the representative locations on the DEM map"""
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import os
        
        # Create visualization directory if it doesn't exist
        vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot DEM as background with terrain colormap
        plt.imshow(self.dem_map, cmap='terrain', alpha=0.7)
        dem_colorbar = plt.colorbar(label='Elevation (m)')
        
        # Create a mask where RLs are located
        y_coords, x_coords = np.where(self.cluster_rl_map == 1)
        
        # Plot RL points in contrasting color
        plt.scatter(x_coords, y_coords, c='red', s=30, marker='o', label=f'RLs (Group {self.rl_group})')
        
        # Add title and labels
        plt.title(f'Representative Locations - Cluster Group {self.rl_group} (n={len(x_coords)})')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.legend(loc='upper right')
        
        # Save the plot
        filename = os.path.join(vis_dir, f'rl_cluster_group_{self.rl_group}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization of {len(x_coords)} representative locations to {filename}")

    def prepare_batch_idxs(self):
        #Create batches separately for each event
        self.train_idx_by_event = {}
        
        for event_id in self.train_event_ids:
            if event_id in self.train_sequences_by_event:
                self.train_idx_by_event[event_id] = self.index_expander_func("train", epoch=0, event_id=event_id)
        
        #Combine all batches from different events
        all_batches = []
        for event_id, batches in self.train_idx_by_event.items():
            for batch in batches:
                # Add event_id info to each batch
                all_batches.append((event_id, batch))
        
        # Shuffle the order of batches between events (use fixed seed for reproducibility)
        rng = np.random.default_rng(42)  # Fixed seed for reproducible results
        rng.shuffle(all_batches)
        
        self.train_idx = all_batches
        logger.info(f"Created {len(self.train_idx)} training batches across {len(self.train_sequences_by_event)} events")
        
        # Create validation batches
        self.validation_idx = []
        for event_id in self.val_event_ids:
            if event_id in self.val_sequences_by_event:
                val_batches = self.index_expander_func("val", epoch=0, event_id=event_id)
                for batch in val_batches:
                    self.validation_idx.append((event_id, batch))
        
        logger.info(f"Created {len(self.validation_idx)} validation batches")
        
    def index_expander_func(self, subset, epoch=0, event_id=None):
        """Create batch indices for a specific event (or all events if event_id is None)"""
        if event_id is not None:
            # Get sequences for this specific event
            if subset == "train":
                if event_id not in self.train_sequences_by_event:
                    return np.array([])
                sequences = self.train_sequences_by_event[event_id]
                all_indices = np.arange(len(sequences))
            elif subset == "val":
                if event_id not in self.val_sequences_by_event:
                    return np.array([])
                sequences = self.val_sequences_by_event[event_id]
                all_indices = np.arange(len(sequences))
            else:
                raise ValueError(f"Invalid subset: {subset}")
        else:
            # Fall back to using all sequences combined (not recommended)
            if subset == "train":
                all_indices = np.arange(len(self.train_sequences))
            elif subset == "val":
                all_indices = np.arange(len(self.val_sequences))
        
        
        rng = np.random.default_rng(42)  # Fixed seed for reproducible results
        rng.shuffle(all_indices)
        
        # Create batches from indices (all from the same event)
        remainder = len(all_indices) % self.batch_size
        # if remainder > 15 self.batch_size add extra batch by adding extra indices
        if remainder >= 10:
            extra_needed = self.batch_size - remainder
            extra_indices = rng.choice(all_indices, size=extra_needed, replace=True)
            all_indices = np.concatenate([all_indices, extra_indices])
            
        num_of_batches = len(all_indices) // self.batch_size
        idx_batch_list = []
        
        if event_id == 7:
            logger.info(f"Preparing batches for event {event_id} with {len(all_indices)} sequences")
        
        for i in range(num_of_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(all_indices))
            if batch_end - batch_start < self.batch_size:
                continue
            batch_indices = all_indices[batch_start:batch_end]
            idx_batch_list.append(batch_indices)
                
        if len(idx_batch_list) > 0:
            idx_batch_list = np.array(idx_batch_list)
            if event_id is not None:
                logger.info(f"Created {len(idx_batch_list)} batches for event {event_id} with {len(all_indices)} sequences")
            else:
                logger.info(f"Created {len(idx_batch_list)} mixed-event batches")
            return idx_batch_list
        else:
            if event_id is not None:
                logger.warning(f"No batches created for event {event_id}. Check sequence count and batch size.")
            else:
                logger.error("No batches were created! Check your data and batch size.")
            return np.array([]) 
        
    def input_tensor_prep(self):
        with torch.no_grad():
            # First load and normalize raw inflow data for each event
            raw_inflow_data = {}
            all_train_data = []
            
            # 1. Load all raw data
            for event_id in self.all_event_ids:
                inflow_file = os.path.join(DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
                inflow_data = pd.read_csv(inflow_file)
                
                inflow_data = inflow_data[8:]  # Skip first 8 rows (warm-up period)
                inflow_data = inflow_data.values
                inflow_data = inflow_data[:, 1:] 
                input_arr = np.array(inflow_data)
                
                # Add padding to the input array from the left
                raw_inflow_data[event_id] = input_arr
                
                # Only collect training data for fitting the scaler
                if event_id in self.train_event_ids:
                    all_train_data.append(input_arr)
            
            # Combine all training data
            all_train_data = np.vstack(all_train_data)
            
            # Create and fit a scaler for each feature independently
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(0,1))  # Avoid exact 0 and 1
            self.scaler.fit(all_train_data)
                
            
            # Store sequences by event to avoid mixing events in batches
            self.train_sequences_by_event = {}  # Dictionary to store sequences by event ID
            self.val_sequences_by_event = {}    # Dictionary to store validation sequences by event ID
            self.test_sequences_by_event = {}   # Dictionary to store test sequences by event ID
            self.train_output_sequences_by_event = {}
            
            # Also maintain combined sequences for backward compatibility
            self.train_sequences = []
            for event_id, raw_data in raw_inflow_data.items():
                normalized_data = self.scaler.transform(raw_data)
                normalized_data = normalized_data.astype(np.float32)
                normalized_data = np.clip(normalized_data, 0, 1)
                if event_id in self.test_event_ids:
                    lowest_base_flow = normalized_data[0, :]   
                
                    # Calculate padding needed
                    padding_length = max(0, self.input_seq_length-1)
                
                    # Create smoothh interpolated padding
                    if padding_length > 0:
                        padding = np.zeros((padding_length, normalized_data.shape[1]))
                        normalized_data = np.vstack((padding, normalized_data))
                        normalized_data[0:padding_length, :] = np.linspace(lowest_base_flow, normalized_data[padding_length, :], padding_length)
                    
                    n_samples = (len(normalized_data) - self.input_seq_length) + 1
                    
                    # Create base sequences
                    input_sequences = []
                    for i in range(n_samples):
                        start_idx = i
                        seq = normalized_data[start_idx:start_idx + self.input_seq_length, :]
                        input_sequences.append(seq)
                        
                    input_sequences = np.array(input_sequences)
                    input_sequences = torch.from_numpy(input_sequences).float()             
                    self.test_sequences_by_event[event_id] = input_sequences          
                else:
                    n_samples = (len(normalized_data) - self.input_seq_length) // self.stride + 1
                    
                    # Create base sequences
                    input_sequences = []
                    for i in range(n_samples):
                        start_idx = i * self.stride
                        seq = normalized_data[start_idx:start_idx + self.input_seq_length, :]
                        input_sequences.append(seq)
                        
                    input_sequences = np.array(input_sequences)
                    input_sequences = torch.from_numpy(input_sequences).float()
                    # Get output at the end of each input sequence, accounting for stride
                    output_indices = np.arange(self.input_seq_length - 1, len(normalized_data), self.stride)
                    output_sequences = self.inundation_data_cache[event_id][output_indices[:len(input_sequences)]]

                    # Store sequences by event ID
                    if event_id in self.test_event_ids:
                        self.test_sequences_by_event[event_id] = input_sequences
                        if self.reconstruction_mode:
                            return
                    
                if event_id in self.train_event_ids:
                    # # Apply temporal shifting with edge padding (creates NEW sequences)
                    # augmented_input_seqs, augmented_output_seqs = self.create_temporal_shifted_sequences(
                    #     normalized_data, self.inundation_data_cache[event_id], event_id
                    # )
                    
                    # Combine original and augmented sequences
                    # self.train_sequences_by_event[event_id] = torch.cat([input_sequences, augmented_input_seqs], dim=0)
                    # self.train_output_sequences_by_event[event_id] = torch.cat([output_sequences, augmented_output_seqs], dim=0)
                    self.train_sequences_by_event[event_id] = input_sequences
                    self.train_output_sequences_by_event[event_id] = output_sequences
                    logger.info(f"Event {event_id}: Prepared {len(self.train_sequences_by_event[event_id])} training sequences")
                elif event_id in self.val_event_ids:
                    self.val_sequences_by_event[event_id] = input_sequences
                    
            # Log feature statistics
            logger.info(f"Feature scaling completed. Number of training sequences: {len(self.train_sequences)}")

    def create_temporal_shifted_sequences(self, normalized_data, target_data, event_id):
        """Create temporally shifted sequences with edge padding to avoid duplicates"""
        augmented_input_seqs = []
        augmented_output_seqs = []
        
        # Define shifts (positive = delay, negative = advance)
        shifts = [-3, -2, 2, 3]  # Try different temporal shifts
        
        for shift in shifts:
            # Create shifted sequences with edge padding
            shifted_input_seqs, shifted_targets = self.apply_temporal_shift_with_padding(
                normalized_data, target_data, shift
            )
            
            if len(shifted_input_seqs) > 0:
                augmented_input_seqs.extend(shifted_input_seqs)
                augmented_output_seqs.extend(shifted_targets)
                logger.info(f"Event {event_id}: Created {len(shifted_input_seqs)} sequences with shift={shift}")
        
        if len(augmented_input_seqs) > 0:
            return torch.stack(augmented_input_seqs), torch.stack(augmented_output_seqs)
        else:
            # Return empty tensors if no augmented sequences created
            empty_input = torch.empty(0, self.input_seq_length, normalized_data.shape[1])
            empty_output = torch.empty(0, target_data.shape[1])
            return empty_input, empty_output

    def apply_temporal_shift_with_padding(self, normalized_data, target_data, shift):
        """Apply temporal shift with edge value padding to create new unique sequences"""
        shifted_input_seqs = []
        shifted_targets = []
        
        original_length = len(normalized_data)
        
        if shift > 0:
            # POSITIVE SHIFT (Delay): Pad at the beginning with first values
            first_values = np.repeat(normalized_data[0:1, :], shift, axis=0)  # Use np.repeat for numpy arrays
            shifted_data = np.vstack([first_values, normalized_data])
            
            # Targets also need to be aligned - use first target values for padding
            if target_data.ndim > 1:
                first_target = target_data[0:1, :].repeat(shift, 1)  # PyTorch tensor repeat
            else:
                first_target = target_data[0:1].repeat(shift)  # 1D tensor repeat
            shifted_target_data = torch.cat([first_target, target_data], dim=0)
            
        elif shift < 0:
            # NEGATIVE SHIFT (Advance): Pad at the end with last values
            last_values = np.repeat(normalized_data[-1:, :], -shift, axis=0)  # Use np.repeat for numpy arrays
            shifted_data = np.vstack([normalized_data, last_values])
            
            # Targets: pad with last target values
            if target_data.ndim > 1:
                last_target = target_data[-1:, :].repeat(-shift, 1)  # PyTorch tensor repeat
            else:
                last_target = target_data[-1:].repeat(-shift)  # 1D tensor repeat
            shifted_target_data = torch.cat([target_data, last_target], dim=0)
            
        else:
            # No shift - return empty (shouldn't happen with our shift values)
            return [], []
        
        # Create sequences from shifted data
        shifted_length = len(shifted_data)
        n_samples = shifted_length - self.input_seq_length + 1
        
        # Only create sequences that don't exactly match original sequences
        for i in range(n_samples):
            seq = shifted_data[i:i + self.input_seq_length, :]
            
            # Check if this sequence would be identical to any original sequence
            is_duplicate = False
            if shift > 0:
                # For positive shifts, check if we're just duplicating original sequences
                original_start_idx = i - shift
                if 0 <= original_start_idx < original_length - self.input_seq_length + 1:
                    original_seq = normalized_data[original_start_idx:original_start_idx + self.input_seq_length, :]
                    if np.allclose(seq, original_seq, rtol=1e-10):
                        is_duplicate = True
            
            # Only add if it's not a duplicate and we have corresponding target
            if not is_duplicate and i < len(shifted_target_data):
                shifted_input_seqs.append(torch.from_numpy(seq).float())
                shifted_targets.append(shifted_target_data[i])
    
        return shifted_input_seqs, shifted_targets
    
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