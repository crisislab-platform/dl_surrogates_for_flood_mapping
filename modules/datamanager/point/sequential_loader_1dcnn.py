import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray, read_shp_point, coords2rc, gdal_transform
import torch
import logging
from modules.utils.run_util import check_device
from modules.datamanager.datamanager import DataManager, check_inundation_data_cache
from modules.datamanager.raster.raster_loader_usrr import ReconsturctionDataManager
import rasterio as rio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataManager")

class CNNSequentialDataManager(DataManager):
    def __init__(self, batch_size=32, input_time_len_h=1, rl_group=1,sampling_dist=20, num_of_clusters=100, fold=1, tuning_mode=True, reconstruction_mode=False, reco_data_manager:ReconsturctionDataManager=None):
        super().__init__()
        # Directories
        self.simulation_data_dir = SIMULATION_DATA_DIR
        self.dem_file = os.path.join(self.simulation_data_dir, "Carlisle_5m.asc")
        self.reconstruction_mode = reconstruction_mode
        self.tuning_mode = tuning_mode
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
        
        # RL variables
        self.rl_group_size = None
        self.rl_group = rl_group
        self.rl_filter = None
        self.sampling_dist = sampling_dist
        self.num_of_clusters = num_of_clusters
        self.rl_cluster_file = self.find_cluster_file(self.rl_group, self.sampling_dist, self.num_of_clusters)

        # Prepare the filter mask for representative locations considered by the RL group in the clusters
        self.prepare_rl_filter(self.rl_group)
        self.event_batch_map = {}
        self.inundation_data_cache = {}
        self.preload_inundation_data()   
        self.input_tensor_prep()
        
        if not self.tuning_mode:
            self.test_input, self.test_output = zip(*self.test_sequences)
            self.test_input = torch.stack(self.test_input, dim=0).float().cuda()
            self.test_output = torch.stack(self.test_output, dim=0).float().cuda()
            
        if not self.reconstruction_mode:
            self.prepare_batch_idxs()
        else:
            self.inundation_data_cache[self.test_event_ids[0]] = self.reco_data_manager.preloaded_maps
            

    def preload_inundation_data(self):
        with torch.no_grad():
            """Preload all inundation data for an event into GPU memory"""
            for event_id in self.all_event_ids:
                if check_inundation_data_cache(event_id):
                    inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{event_id}_inundation.pt")).cuda()
                    event_data = []
                    for i in range(len(inundation_data)):
                        t_data = inundation_data[i]
                        filtered_tensor = self.rl_filter(t_data)
                        event_data.append(filtered_tensor)
                    self.inundation_data_cache[event_id] = torch.stack(event_data, dim=0)
                    logger.info(f"inundation cache size:  {self.inundation_data_cache[event_id].shape}")
                    logger.info(f"Loaded inundation data for event {event_id} from cache")
                else:
                    raise ValueError(f"Inundation data for event {event_id} not found in cache. Please run create_inundation_map_tensors() first.")
        
    def get_batch(self, indices, subset="train"):
        if subset == "train":
            sequences = [self.train_sequences[idx] for idx in indices]
        elif subset == "val":
            sequences = [self.val_sequences[idx] for idx in indices]
        else:
            raise ValueError(f"Invalid subset: {subset}. Choose 'train' or 'val'.")
        
        input_tensors, output_tensors = zip(*sequences)
            
        input_tensor = torch.stack(input_tensors, dim=0).float().cuda()
        output_tensor = torch.stack(output_tensors, dim=0).float().cuda()
        return input_tensor, output_tensor
        
    def prepare_rl_filter(self, rl_group):
        self.rl_filter = None
        self.cluster_rls = self.find_cluster_rls(self.rl_cluster_file, rl_group)
        self.rl_group_size = len(self.cluster_rls)
        self.dem_map = gdal_asarray(self.dem_file)
        self.cluster_rl_map = np.zeros(self.dem_map.shape)
        for row, col in self.cluster_rls:
            self.cluster_rl_map[row, col] = 1
       
        self.filter_mask  = self.cluster_rl_map == 1
        self.rl_filter = lambda x: x[self.filter_mask]
        logger.info(f"RL filter prepared.")
    
        # I need to visualize the representative locations on the DEM
        self.visualize_rls_cluster()
        
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
        idx_expander = lambda x: self.index_expander_func(x)
        self.train_idx = idx_expander("train")
        # self.val_idx = idx_expander("val")
        # logger.info(f"Created {len(self.train_idx)} training batches and {len(self.val_idx)} validation batches")

    def index_expander_func(self, subset):
        if subset == "train":
            all_indices = np.arange(len(self.train_sequences))
        elif subset == "val":
            all_indices = np.arange(len(self.val_sequences))
            
        rng = np.random.default_rng(341)
        rng.shuffle(all_indices)
        
        # Create batches from mixed indices
        num_of_batches = len(all_indices) // self.batch_size
        idx_batch_list = []
        
        for i in range(num_of_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(all_indices))
            if batch_end - batch_start < self.batch_size:
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
        
    def input_tensor_prep(self):
        with torch.no_grad():
            # First load and normalize raw inflow data for each event
            raw_inflow_data = {}
            all_train_data = []
            
            # 1. Load all raw data
            for event_id in self.all_event_ids:
                inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
                inflow_data = pd.read_csv(inflow_file)
                
                inflow_data = inflow_data.iloc[8:,:]
                inflow_data = inflow_data.values
                inflow_data = inflow_data[:, 1:] 
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
            
            # # Create and fit a scaler for each feature independently
            self.scaler = MinMaxScaler(feature_range=(0,1)) # Use StandardScaler for zero mean and unit variance
            self.scaler.fit(all_train_data)  # Fit on all training data - this automatically scales each feature independently
            
            self.train_sequences = []
            
            for event_id, raw_data in raw_inflow_data.items():
                # Apply feature-wise normalization
                normalized_data = self.scaler.transform(raw_data)
                normalized_data = np.r_['0,2', np.zeros((self.input_seq_length-1, normalized_data.shape[1])), normalized_data]
                normalized_data[:self.input_seq_length-1, :] = np.repeat(normalized_data[self.input_seq_length-1:self.input_seq_length, :], self.input_seq_length-1, axis=0)  # Fill padding with first valid values
                n_samples = len(normalized_data) - self.input_seq_length + 1
                
                # Create sequences
                input_sequences = [normalized_data[i: i + self.input_seq_length, :] for i in range(n_samples)]
                input_sequences = np.array(input_sequences)
                
                # Convert to tensors - explicitly specify torch.float32 to match model parameters
                input_sequences = torch.from_numpy(input_sequences).cuda()
                outputs = self.inundation_data_cache[event_id].clone() 
                # Clone to avoid modifying the original data
                
                sample_input = input_sequences
                sample_output = outputs

                logger.info(f"Event {event_id}:")
                logger.info(f"  Input shape: {sample_input.shape}")
                logger.info(f"  Input range: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
                logger.info(f"  Input std: {sample_input.std():.3f}")
                logger.info(f"  Output shape: {sample_output.shape}")
                logger.info(f"  Output range: [{sample_output.min():.3f}, {sample_output.max():.3f}]")
                logger.info(f"  Output non-zero %: {(sample_output > 0).float().mean():.3f}")
                
                # Check for data issues
                if sample_input.std() < 0.01:
                    logger.warning("  ⚠️  Very low input variance - scaling issue?")
                if (sample_output > 0).float().mean() < 0.01:
                    logger.warning("  ⚠️  Very sparse outputs - most locations never flood")
                        
                # Check for shape mismatches
                if outputs.shape[0] != input_sequences.shape[0]:
                    logger.error(f"Mismatch in output length for event {event_id}. Expected: {n_samples}, Found: {len(outputs)}")
                    raise ValueError(f"Mismatch in output length for event {event_id}. Expected: {n_samples}, Found: {len(outputs)}")
            
                # Create a pair of input and output sequences ((input_tensor, output_tensor), ...)
                sequence_pairs = list(zip(input_sequences, outputs))

                if event_id in self.train_event_ids:
                    self.train_sequences.extend(sequence_pairs)
                if event_id in self.test_event_ids:
                    self.test_sequences = sequence_pairs
                if event_id in self.val_event_ids:
                    self.val_sequences = sequence_pairs
                    
            # Log feature statistics
            logger.info(f"Feature scaling completed. Number of training sequences: {len(self.train_sequences)}")
        
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
    
    def reshuffle_train_indices(self):
        if hasattr(self, "train_idx") and len(self.train_idx) > 0:
            flat = self.train_idx.ravel()
            rng = np.random.default_rng()
            rng.shuffle(flat)
            self.train_idx = flat.reshape(-1, self.batch_size)