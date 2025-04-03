import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_asarray, read_shp_point, coords2rc, gdal_transform
import torch
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNNDataManager")

class CNNDataManager():
    def __init__(self, batch_size, input_time_len_h, time_lag_h, rl_group, timestep=1, sampling_dist=20, num_of_clusters=100,test_mode=False):
        
        self.test_mode = test_mode
        
        # Squence variables
        self.batch_size = batch_size
        self.timestep = timestep # Time step multipler against the original data (15 min * multipler)
        self.time_lag_h = 0
        self.event_seq_data = {}
        
        self.input_seq_length = int(input_time_len_h * 4 / timestep) 
        self.input_seq_start = int((self.time_lag_h * 4) / timestep) + self.input_seq_length
    
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

        # Baseline flow rates for upstream1, upstream2, and upstream3
        self.up1_baseline_fr = 69.2354
        self.up2_baseline_fr = 6.2953
        self.up3_baseline_fr = 14.7835
        
        #directories
        self.simulation_dir = SIMULATION_DATA_DIR
        self.dem_asc_file = os.path.join(CARLISLE_DATA_DIR, "Carlisle_5m.asc")
        
        self.rl_filter_prepare(self.rl_group)
        if not self.test_mode:
            self.train_loader, self.val_loader = self.get_batch_rl_loaders()
        else:
            self.test_inputs = self.input_tensor_prep()
            

    def input_tensor_prep(self):
        event_input = []
        for event_id in self.all_event_ids:
            inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            # Exclude first 8 rows
            inflow_data = inflow_data.iloc[8:]
            inflow_data = inflow_data.reset_index(drop=True)
            
            upstream1_idx = inflow_data.columns.to_list().index("Upstream1")
            upstream2_idx = inflow_data.columns.to_list().index("Upstream2")
            upstream3_idx = inflow_data.columns.to_list().index("Upstream3")
            
            input_arr = np.array(inflow_data)
            input_arr = self.time_slicing(input_arr[:,1:]) #remove time column and slice the data

            zero_padding = np.zeros((self.input_seq_start-1, input_arr.shape[1]))
            input_arr = np.r_['0,2', zero_padding, input_arr]
            
            # Set a baseline flow rate for padded values to complete the first sequence
            input_arr[:self.input_seq_start-1, upstream1_idx-1] = self.up1_baseline_fr
            input_arr[:self.input_seq_start-1, upstream2_idx-1] = self.up2_baseline_fr
            input_arr[:self.input_seq_start-1, upstream3_idx-1] = self.up3_baseline_fr
        
            #create sequences of length seq_len
            event_input.append([input_arr[i:i+self.input_seq_length, :] for i in range(len(self.time_slicing(inflow_data)))])
             
        # current shape = (event_num, evt_data_points, seq_len, var_num) and inhomogeneous
        event_input = np.concatenate(event_input, axis=0) #shape = (event_num * evt_data_points, seq_len, var_num)
        logger.info(f"Event input shape after conversion: {event_input.shape}")
        event_input = event_input.reshape((-1, *event_input.shape[-2:]))
        logger.info(f"Input data shape after reshaping: {event_input.shape}")
        
        # Scale the data to the range [0, 1], skip scaling for the moment
        #event_input = event_input/ 1726.462853 #max value of inflow data
        return torch.from_numpy(event_input).float()   
    
    def get_batch_rl_loaders(self, shuffle=True, random_seed=321):
        train_loader, validation_loader = self.idx_prep(self.batch_size,  shuffle, random_seed)
        logger.info(f"Train loader shape: {len(train_loader)}")
        logger.info(f"Validation loader shape: {len(validation_loader)}")
        return train_loader, validation_loader
        
    def get_no_time_steps(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_dir}/Run{event_id}-*.wd")
        return len(inundation_files) - 8
    
    def index_expander_func(self, x):
        idx_per_event = lambda x: np.ceil((self.get_no_time_steps(x) - self.timestep//2) / self.timestep).astype(int)
        idx_list = []
        
        even_start_id_map = {}
        for index, event_id in enumerate(self.all_event_ids):
            if index == 0:
                start_idx = 0
            else:
                start_idx = even_start_id_map[self.all_event_ids[index-1]] + idx_per_event(self.all_event_ids[index-1])
            even_start_id_map[event_id] = start_idx
                
        # Should also consider event_id
        for event_id in x:
            #Find start_idx
            start_idx = even_start_id_map[event_id]
            num_idx = idx_per_event(event_id)
            # Generate sequential indices starting from start_idx
            idx = start_idx + np.arange(num_idx)
            idx_list.append(idx)
            # Update start_idx for the next event
            start_idx += num_idx
        idx_list = np.concatenate(idx_list)
        logger.info(f"Index list shape: {idx_list.shape}")
        return idx_list
    
    def idx_prep(self,batch_size, shuffle=True, random_seed=341):
        logger.info(f"Preparing index for batch size: {batch_size}")
        inputs_tensor = self.input_tensor_prep()
        rls_ts = self.rl_dataset_prep()
        tensor_ls = [inputs_tensor, rls_ts]
        
        # given a set of event_ids(x), create a sequence of samples
        # for each event_id, create a sequence of samples
        # each event has a different number of steps therefore need to calculate the number of steps

        idx_expander = lambda x: self.index_expander_func(x)
        train_map_idxs = idx_expander([i for i in self.all_event_ids if i not in self.test_event_ids and i not in self.validation_event_ids])
        validation_map_idxs = idx_expander(self.validation_event_ids)
        
        logger.info(f"Train map indices shape: {train_map_idxs.shape} Max: {train_map_idxs.max()}")
        logger.info(f"Validation map indices shape: {validation_map_idxs.shape} Max: {validation_map_idxs.max()}")
        logger.info(f"Input tensor shape: {inputs_tensor.shape}")
        logger.info(f"RL tensor shape: {rls_ts.shape}")
        
        rng = np.random.default_rng(random_seed)
        if shuffle:
            rng.shuffle(train_map_idxs)
            rng.shuffle(validation_map_idxs)
            
        train_data_ls  = [torch.split(tensor_i[train_map_idxs.astype(int)], batch_size) for tensor_i in tensor_ls]
        validation_data_ls = [torch.split(tensor_i[validation_map_idxs.astype(int)], batch_size) for tensor_i in tensor_ls]
        
        if len(tensor_ls) > 1:
            train_loader = list(zip(*train_data_ls))
            test_loader = list(zip(*validation_data_ls))
        else:
            train_loader = train_data_ls[0]
            test_loader = validation_data_ls[0]
        logger.info("Train and validation loaders prepared.")
        return train_loader, test_loader
        
    def read_event_inundation_data(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_dir}/Run{event_id}-*.wd")
        inundation_files.sort()
        inundation_files = inundation_files[8:]  # Skip the first 8 rows
        inundation_data_arr = []
    
        for t_idx, inundation_file in enumerate(inundation_files):
            inundation_data = gdal_asarray(inundation_file)
            logger.info(f"Inundation data shape before filtering: {inundation_data.shape}")
            timestep_rl_inundation = self.rl_filter(inundation_data)
            logger.info(f"Inundation data shape after filtering: {timestep_rl_inundation.shape}")
            logger.info(f"sample inundation data: {timestep_rl_inundation[0]}")
            inundation_data_arr.append(timestep_rl_inundation)
        
        logger.info(f"Inundation data shape: {np.array(inundation_data_arr).shape}")
        inundation_data_arr = np.array(inundation_data_arr) 
        # Now the shape is (time_steps, num_points)
        inundation_data_arr = self.time_slicing(inundation_data_arr)
        logger.info(f"Inundation data shape after slicing: {inundation_data_arr.shape}")
        return inundation_data_arr
        
    def rl_dataset_prep(self):
        # initialise the array with first event
        rl_curr = self.read_event_inundation_data(self.all_event_ids[0])
        dataset_arr = rl_curr
        logger.info(f"RL curr data shape: {rl_curr.shape}")
        logger.info(f"Dataset shape after initialisation: {dataset_arr.shape}")
        if len(self.all_event_ids) > 1:
            for i in range(1, len(self.all_event_ids)):
                rl_curr = self.read_event_inundation_data(self.all_event_ids[i])
                logger.info(f"RL curr data shape: {rl_curr.shape}")
                dataset_arr = np.append(dataset_arr, rl_curr, axis=0)
                logger.info(f"Dataset shape after appending: {dataset_arr.shape}")
           
        logger.info(f"RL dataset shape: {np.array(dataset_arr).shape}")
        return torch.from_numpy(dataset_arr)
        
    def find_cluster_file(self, rl_group, sampling_dist, num_of_clusters):
        if rl_group is None:
            return None, None
        cluster_file = f"{OUTPUT_DIR}/rls/clusters/cluster_{rl_group}_ss_{sampling_dist}_{num_of_clusters}.shp"
        if not os.path.exists(cluster_file):
            logger.error(f"Cluster file {cluster_file} does not exist.")
            raise ValueError(f"Cluster file {cluster_file} does not exist.")
        return cluster_file
        
    def rl_filter_prepare(self, rl_group):
        logger.info(f"Preparing RL filter for group {rl_group}")
        if rl_group is None:
            self.rl_filter = lambda x: x
            return 0
        self.cluster_rls= read_shp_point(self.rl_cluster_file)
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


