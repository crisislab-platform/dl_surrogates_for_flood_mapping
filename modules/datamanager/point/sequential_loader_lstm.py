import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray, read_shp_point, coords2rc, gdal_transform
import torch
import logging
import glob
from modules.utils.run_util import check_device
from modules.datamanager.datamanager import DataManager, check_inundation_data_cache
from modules.datamanager.raster.raster_loader_usrr import ReconsturctionDataManager
import rasterio as rio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTMSRRDataManager")

class LSTMSequentialDataManager(DataManager):
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
            self.val_event_ids = [1]
            
            
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
        
        self.inundation_data_cache = {}
        if not self.reconstruction_mode:
            self.prepare_batch_idxs()
        else:
            self.inundation_data_cache[self.test_event_ids[0]] = self.reco_data_manager.preloaded_maps
            
        if not self.tuning_mode:
            test_sequences = self.test_sequences[self.test_start_index:self.test_end_index + 1]
            #test_sequences is a tuple of ((input_tensor, output_tensor), ...)
            test_input = []
            test_output = []
            for input_tensor, output_tensor in test_sequences:
                test_input.append(input_tensor)
                test_output.append(output_tensor)
            test_input = torch.stack(test_input, axis=0)
            test_output = torch.stack(test_output, axis=0)
            self.test_input = test_input.cuda()
            self.test_output = test_output.cuda()
            
    def preload_inundation_data(self):
        """Preload all inundation data for an event into GPU memory"""
        for event_id in self.all_event_ids:
            if check_inundation_data_cache(event_id):
                inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{event_id}_inundation.pt"))
                self.inundation_data_cache[event_id] = []
                for i in range(len(inundation_data)):
                    tensor = inundation_data[i].cuda() 
                    filtered_tensor = self.rl_filter(tensor)
                    # if event_id == 1 and torch.max(filtered_tensor).item() != 0:
                    #     raise ValueError(f"Filtered inundation tensor for event {event_id} contains non-zero values. Check your RL filter.")
                    self.inundation_data_cache[event_id].append(filtered_tensor)
                logger.info(f"inundation cache size:  {inundation_data.shape}")
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
            
        input_tensor = torch.stack(input_tensors, dim=0).cuda()
        output_tensor = torch.stack(output_tensors, dim=0).cuda()
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

    def get_representative_locations(self):
        rl_list = pd.read_csv(self.rep_loc_file_path)
        logger.info(f"rep loc: {rl_list.iloc[0]}")
        rl_list = rl_list[['x', 'y']].values
        rl_list = [tuple(x) for x in rl_list]
        
        transform = gdal_transform(self.dem_asc_file)
        dem_map_np = gdal_asarray(self.dem_asc_file)

        # Convert RL coordinates to row/col indices
        self.rc_rl_points = [coords2rc(transform, rl) for rl in rl_list] 
        transform = gdal_transform(self.dem_asc_file)
        dem_map_np = gdal_asarray(self.dem_asc_file)
        
        # Convert RL coordinates to row/col indices
        rc_rl_points = [coords2rc(transform, rl) for rl in  rl_list] 
        rl_map_np = np.zeros(dem_map_np.shape)
        for row, col in rc_rl_points:
            rl_map_np[row, col] = 1
        
        logger.info(f"Loaded {len(rl_list)} representative locations ")
        return rl_map_np, dem_map_np

    def prepare_batch_idxs(self):
        idx_expander = lambda x: self.index_expander_func(x)
        self.train_idx = idx_expander("train")
        self.val_idx = idx_expander("val")
        logger.info(f"Created {len(self.train_idx)} training batches and {len(self.val_idx)} validation batches")

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
        # First load and normalize raw inflow data for each event
        raw_inflow_data = {}
        all_train_data = []
        
        # 1. Load all raw data
        for event_id in self.all_event_ids:
            inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            # Sort by time
            inflow_data = inflow_data.sort_values(by='Time')
            inflow_data = inflow_data.iloc[8:,:] # Skip the first 8 rows
            inflow_data = inflow_data.values
            inflow_data = inflow_data[:, 1:] # Skip the first column
            input_arr = np.array(inflow_data.astype(np.float32))
            
            # Add rate of change feature
            input_arr = np.r_['0,2', np.zeros((self.input_seq_length-1, input_arr.shape[1])), input_arr]
            input_arr[:self.input_seq_length-1, :] = np.repeat(input_arr[self.input_seq_length-1:self.input_seq_length, :], self.input_seq_length-1, axis=0)  # Fill padding with first valid values
            raw_inflow_data[event_id] = input_arr
            
            # Only collect training data for fitting the scaler
            if event_id in self.train_event_ids:
                all_train_data.append(input_arr)
        
        # 2. Normalize the data using MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit the scaler on all training data
        all_train_data = np.vstack(all_train_data)
        scaler.fit(all_train_data)  # Fit on all training data
        
        self.train_sequences = []
        for event_id, raw_data in raw_inflow_data.items():
            normalized_data = scaler.transform(raw_data) # Normalize by max inflow
            n_samples = len(normalized_data) - self.input_seq_length + 1
            # input_sequences = np.zeros((n_samples, self.input_seq_length, n_features))
    
            input_sequences = [normalized_data[i: i + self.input_seq_length, :] for i in range(n_samples)]
            input_sequences = np.array(list(input_sequences)) 
            
            outputs = self.inundation_data_cache[event_id]
            
            input_sequences = torch.from_numpy(input_sequences).float().cuda()
            outputs = torch.stack(outputs, dim=0).cuda()
            
            if outputs.shape[0] != input_sequences.shape[0] and event_id not in self.test_event_ids:
                logger.error(f"Mismatch in output length for event {event_id}. Expected: {n_samples}, Found: {len(outputs)}")
                raise ValueError(f"Mismatch in output length for event {event_id}. Expected: {n_samples}, Found: {len(outputs)}")
           
            #Create a pair of input and output sequences ((input_tensor, output_tensor), ...)
            sequence_pairs = []
            for i in range(input_sequences.shape[0]):
                # Extract one sample (time sequence) from input and its corresponding output
                sample_input = input_sequences[i] 
                sample_output = outputs[i]
                sequence_pairs.append((sample_input, sample_output))

            if event_id in self.train_event_ids:
                self.train_sequences.extend(sequence_pairs)
            if event_id in self.test_event_ids:
                self.test_sequences = sequence_pairs
            if event_id in self.val_event_ids:
                self.val_sequences = sequence_pairs
                
        # Concatenate all training sequences into a single array
        logger.info("Completed nmalization and sequence creation")
        
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