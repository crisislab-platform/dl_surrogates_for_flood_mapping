import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_asarray, read_shp_point, coords2rc, gdal_transform
import torch
import logging
import glob
from modules.utils.run_util import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataManager")

class CNNDataManager():
    def __init__(self, batch_size=32, input_time_len_h=1, time_lag_h=0, rl_group=1, timestep=1, sampling_dist=20, num_of_clusters=100,test_mode=False):
        
        self.test_mode = test_mode
        
        # Squence variables
        self.batch_size = batch_size
        self.timestep = timestep # Time step multipler against the original data (15 min * multipler)
        self.time_lag_h = 0 # Time lag in hours
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

        # Event IDs for training, testing, and validation
        self.all_event_ids = [1,2,3,4,5,6,7,8,9]
        self.test_event_ids = [1]
        self.validation_event_ids = [9]
        
        if self.test_mode:
            self.all_event_ids = self.test_event_ids
            self.validation_event_ids = []

        #directories
        self.simulation_dir = SIMULATION_DATA_DIR
        self.dem_asc_file = os.path.join(self.simulation_dir, "Carlisle_5m.asc")
        
        self.rl_filter_prepare(self.rl_group)
        
        self.idx_per_event = lambda x: np.ceil((self.get_no_time_steps(x) - self.timestep//2) / self.timestep).astype(int)
        even_start_id_map = {}
        for index, event_id in enumerate(self.all_event_ids):
            if index == 0:
                start_idx = 0
            else:
                start_idx = even_start_id_map[self.all_event_ids[index-1]] + self.idx_per_event(self.all_event_ids[index-1])
            even_start_id_map[event_id] = start_idx
        self.event_start_id_map = even_start_id_map
        self.train_loader, self.val_loader, self.test_inputs = self.get_batch_rl_loaders()

    def input_tensor_prep(self):
        event_input = []
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
            event_input.append(event_sequences)
        event_input = np.concatenate(event_input, axis=0)
        event_input = event_input.reshape((-1, self.input_seq_length, event_input.shape[-1]))
        event_input  = event_input
        result = torch.from_numpy(event_input).float()
        return result   
    
    def get_batch_rl_loaders(self, shuffle=True, random_seed=321):
        train_loader, validation_loader, test_inputs = self.data_prep(self.batch_size,  shuffle, random_seed)
        return train_loader, validation_loader, test_inputs
        
    def get_no_time_steps(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_dir}/Run{event_id}-*.wd")
        return len(inundation_files) - 8 - self.input_seq_start
    
    def index_expander_func(self, event_ids):
        idx_list = []
        if len(event_ids) == 0:
            return np.array([])
        for event_id in event_ids:
            start_idx = self.event_start_id_map[event_id]
            num_idx = self.idx_per_event(event_id)
            idx = start_idx + np.arange(num_idx)
            idx_list.append(idx)
        idx_list = np.concatenate(idx_list)
        logger.info(f"Index list shape: {idx_list.shape}")
        return idx_list
    
    def data_prep(self, batch_size, shuffle=True, random_seed=341):
        logger.info(f"Preparing index")
        idx_expander = lambda x: self.index_expander_func(x)
        train_map_idxs = idx_expander([i for i in self.all_event_ids if i not in self.test_event_ids and i not in self.validation_event_ids])
        validation_map_idxs = idx_expander(self.validation_event_ids)
        test_map_idxs = idx_expander(self.test_event_ids)
        inputs_tensor = self.input_tensor_prep()
        rls_ts = self.rl_dataset_prep()
        tensor_ls = [inputs_tensor, rls_ts]
        

        logger.info(f"Input tensor shape: {inputs_tensor.shape}")
        logger.info(f"RL tensor shape: {rls_ts.shape}")
        
        if not self.test_mode:
            logger.info(f"Train map indices shape: {train_map_idxs.shape} Max: {train_map_idxs.max()}")
            logger.info(f"Validation map indices shape: {validation_map_idxs.shape} Max: {validation_map_idxs.max()}")
            rng = np.random.default_rng(random_seed)
            if shuffle:
                rng.shuffle(train_map_idxs)
                rng.shuffle(validation_map_idxs)
        
        # Convert indices to tensors
        if not self.test_mode:
            train_indices = torch.tensor(train_map_idxs.astype(int))
            validation_indices = torch.tensor(validation_map_idxs.astype(int))
        test_indices = torch.tensor(test_map_idxs.astype(int))
        
        train_data_ls = []
        validation_data_ls = []
        test_data_ls = []
        for tensor_i in tensor_ls:
            if self.test_mode:
                test_tensor = tensor_i[test_indices]
                test_data_ls.append(test_tensor)
                continue
            train_tensor = tensor_i[train_indices]
            validation_tensor = tensor_i[validation_indices]
            train_batches = list(torch.split(train_tensor, batch_size))
            validation_batches = list(torch.split(validation_tensor, batch_size))
            train_data_ls.append(train_batches)
            validation_data_ls.append(validation_batches)
 
        if self.test_mode:
            return None, None, test_data_ls
        train_loader = list(zip(*train_data_ls))
        valdiation_loader = list(zip(*validation_data_ls))
        logger.info("Train and validation loaders prepared and moved to GPU.")
        return train_loader,valdiation_loader, test_data_ls
        
    def read_event_inundation_data(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_dir}/Run{event_id}-*.wd")
        inundation_files.sort()
        inundation_files = inundation_files[8:]  #Skip the first 8 rows
        inundation_data_arr = []
        
        for t_idx, inundation_file in enumerate(inundation_files):
            if t_idx < self.input_seq_start:
                continue
            inundation_data = gdal_asarray(inundation_file)
            timestep_rl_inundation = self.rl_filter(inundation_data)
            inundation_data_arr.append(timestep_rl_inundation)
        
        inundation_data_arr = np.array(inundation_data_arr) 
        inundation_data_arr = self.time_slicing(inundation_data_arr)
        return inundation_data_arr
        
    def rl_dataset_prep(self):
        rl_curr = self.read_event_inundation_data(self.all_event_ids[0])
        dataset_arr = rl_curr
        if len(self.all_event_ids) > 1:
            for i in range(1, len(self.all_event_ids)):
                rl_curr = self.read_event_inundation_data(self.all_event_ids[i])
                dataset_arr = np.append(dataset_arr, rl_curr, axis=0)
        logger.info(f"RL dataset shape: {np.array(dataset_arr).shape}")
        result = torch.from_numpy(dataset_arr)
        return result
        
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
        
    def rl_filter_prepare(self, rl_group):
        logger.info(f"Preparing RL filter for group {rl_group}")
        if rl_group is None:
            self.rl_filter = lambda x: x
            return 0
        self.cluster_rls = self.find_cluster_rls(self.rl_cluster_file, rl_group)
        self.rl_group_size = len(self.cluster_rls)
        transform  = gdal_transform(self.dem_asc_file)
        self.dem_map = gdal_asarray(self.dem_asc_file)
        self.coords_to_cluster_rls = [coords2rc(transform, coord)  for coord in self.cluster_rls]
        
        self.cluster_rl_map = np.zeros(self.dem_map.shape)
        for row, col in self.coords_to_cluster_rls:
            self.cluster_rl_map[row, col] = 1
       
        filter_mask  = self.cluster_rl_map == 1
        self.rl_filter = lambda x: x[filter_mask]
        logger.info(f"RL filter prepared.")
