import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_asarray, read_shp_point, coords2rc, gdal_transform
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
        
        # Directories
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
            
        # Squence variables
        self.batch_size = batch_size
        self.tuning_mode = tuning_mode
        self.timestep = timestep # Time step multipler against the original data (15 min * multipler)
        self.event_seq_data = {}
        self.input_seq_length = int(input_time_len_h * 4 / timestep) 
        self.input_seq_start = self.input_seq_length
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
        if not self.tuning_mode:
            self.test_input = self.event_input_map[self.test_event_ids[0]]
            self.prep_test_ouput_data()
            
    def get_batch(self, indices):
        event_id, local_idxs =  self.find_time_steps(indices)
        if event_id is None or local_idxs is None:
            logger.error(f"Could not find event ID or local indices for index {indices}")
            return None, None
        
        event_data = self.event_input_map[event_id]
        batch_data = np.array([event_data[idx] for idx in local_idxs if idx < len(event_data)])
        
        input_batch = torch.from_numpy(batch_data).float()
        output_batch = self.load_inundation_data(event_id, local_idxs)
        
        return input_batch, output_batch
        
    def prep_test_ouput_data(self):
        time_steps = self.get_no_time_steps(self.test_event_ids[0])
        time_indices = np.arange(time_steps)
        inundation_data = self.load_inundation_data(self.test_event_ids[0], time_indices)
        self.test_output= inundation_data
        logger.info(f"Test output data shape: {self.test_output.shape}")
        
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
            end_idx = start_idx + self.get_no_time_steps(event_id)
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
            event_sequences = [input_arr[i:i+self.input_seq_length, :] for i in range(len(self.time_slicing(inflow_data))-self.input_seq_length)]
            event_sequences = np.array(event_sequences)
            event_sequences = event_sequences.reshape((-1, self.input_seq_length, event_sequences.shape[-1]))
            self.event_input_map[event_id] = event_sequences
            if event_id ==  self.test_event_ids[0]:
                self.test_input = torch.from_numpy(event_sequences)
          
    def get_no_time_steps(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_data_dir}/Run{event_id}-*.wd")
        return len(inundation_files) - 8 - self.input_seq_start
    
    def index_expander_func(self, event_ids):
        idx_batch_list = []
        if len(event_ids) == 0:
            return np.array([])
        
        for event_id in event_ids:
            start_idx = self.event_start_id_map[event_id]
            num_idx = self.idxs_per_event(event_id)
            num_of_batches = num_idx // self.batch_size
            for i in range(num_of_batches):
                batch_start = i * self.batch_size
                batch_end = min((i + 1) * self.batch_size, num_idx)
                if batch_end - batch_start < self.batch_size:
                    # Skip incomplete batches
                    continue
                batch_indices = start_idx + np.arange(batch_start, batch_end)
                idx_batch_list.append(batch_indices)
                
        if len(idx_batch_list) > 0:
            idx_batch_list = np.array(idx_batch_list)
            logger.info(f"Index batches shape: {idx_batch_list.shape}")
            return idx_batch_list
        else:
            logger.error("No batches were created! Check your data and batch size.")
            return np.array([])  #Return empty array as fallback
        
    def load_inundation_data(self, event_id, time_indices):
        event_inundation_files = glob.glob(f"{self.simulation_data_dir}/Run{event_id}-*.wd")
        event_inundation_files.sort()
        event_inundation_files = event_inundation_files[8:]
        
        event_inundation_files = [event_inundation_files[i] for i in time_indices]
        event_inundation_data = []
        for i, file in enumerate(event_inundation_files):
            inundation_data = gdal_asarray(file)
            inundation_data = self.rl_filter(inundation_data)
            event_inundation_data.append(inundation_data)
            
        event_inundation_data = np.array(event_inundation_data)
        event_inundation_data = self.time_slicing(event_inundation_data)
        
        # Set negligible inundation to 0
        event_inundation_data[event_inundation_data < 0.3] = 0
        return torch.from_numpy(event_inundation_data).float()
        
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
    