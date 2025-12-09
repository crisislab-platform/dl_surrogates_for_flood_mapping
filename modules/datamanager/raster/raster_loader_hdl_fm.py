import logging
import os
import numpy as np
import pandas as pd
import rasterio as rio
import glob
from sklearn.preprocessing import MinMaxScaler
import torch
from modules.lib.constants import DATA_DIR, SIMULATION_DATA_DIR, OUTPUT_DIR
from modules.datamanager.datamanager import DataManager
from modules.utils.run_util import check_device
from modules.lib.gdal_lib import gdal_asarray, coords2rc
from modules.datamanager.datamanager import check_inundation_data_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HDLFMataLoader")

class HDLFMRasterDataManager(DataManager):
        
    def __init__(self, lag=8, batch_size=32, validation_event=2, tuning_mode=True):
        super().__init__()
        if tuning_mode:
            self.train_event_ids = [event for event in [2, 3, 4, 5, 6, 7, 8, 9] if event != validation_event]
            self.test_event_ids = [1]
            self.val_event_ids = [validation_event]
        else:   
            self.train_event_ids = [2, 3, 4, 5, 6, 7, 8, 9]
            self.test_event_ids = [1]
            self.val_event_ids = [1]
            
        self.all_event_ids = np.concatenate([self.train_event_ids, self.test_event_ids, self.val_event_ids])
        
        self.simulation_data_dir = SIMULATION_DATA_DIR
        self.dem_file = os.path.join(self.simulation_data_dir, "Carlisle_5m.asc")
        
        #Remove duplicates
        self.all_event_ids = np.unique(self.all_event_ids)
        self.all_event_ids.sort()
        self.lag = lag # No of antecedent time steps
        self.input_seq_length = lag
        self.batch_size = batch_size
        self.train_idx = None
        self.validation_idx = None
        self.test_index = None
        self.device = check_device()
        self.outputs = self.find_output_size()
        self.features = 3  # 3; upstream inputs + DEM + inundation data
        
        self.event_input_map = {}
        self.event_batch_map = {}
        self.inundation_data_cache = {}
        self.upstream_filters = {}
        self.dem_raster = gdal_asarray(self.dem_file)
        self.zero_dem_raster = torch.from_numpy(np.zeros_like(self.dem_raster)).float()
        self.dem_raster = torch.from_numpy(self.dem_raster).float().cuda()
        
        try:
            self.idx_prep()
            self.find_upstream_point_fiters()
            self.inflow_data_prep(self.all_event_ids)
            self.preload_inundation_data(self.all_event_ids)
            self.test_idx_prep()
                  
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise e
            
    
    def preload_inundation_data(self, event_ids):
        """Preload and cache all inundation data for specified event IDs"""
        logger.info(f"Preloading inundation data for events: {event_ids}")
        for event_id in event_ids:
            if check_inundation_data_cache(event_id):
                self.inundation_data_cache[event_id] = []
                inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{event_id}_inundation.pt"))
                for i in range(len(inundation_data)):
                    if torch.cuda.is_available() and self.device != 'cpu':
                        tensor_data = inundation_data[i].cuda()
                        # Fix: Create the zero tensor on the same device
                        zero_tensor = torch.tensor(0.0, device=self.device)
                        tensor_data = torch.where(tensor_data < 0.3, zero_tensor, tensor_data)
                        self.inundation_data_cache[event_id].append(tensor_data)
                    else:
                        self.inundation_data_cache[event_id].append(inundation_data[i])
            else:
                 raise ValueError(f"Inundation data for event {event_id} not found in cache. Please run create_inundation_map_tensors() first.")
            
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
            inflow_data = inflow_data[8:] #Skip the first 8 rows
            inflow_data = inflow_data.dropna()
            flow_data = inflow_data.drop(columns=['Time'])
            input_arr = np.array(flow_data.values.astype(np.float32))
            self.event_input_map[event_id] =  input_arr
               
        #Fit the flow scaler on all training data
        all_train_data = np.vstack([self.event_input_map[id] for id in self.train_event_ids])
        flow_scaler = MinMaxScaler(feature_range=(0, 1))
        flow_scaler.fit(all_train_data)
        
        for event_id in self.all_event_ids:
            scaled_flow = flow_scaler.transform(self.event_input_map[event_id])
            self.event_input_map[event_id] = scaled_flow
            
        
    def get_batch(self, indices, subset="train"):
        event_indices_map = self.find_indices(indices)
        if not event_indices_map:
            logger.error(f"Could not find event IDs or local indices for indices {indices}")
            return None, None
        
        input_data = []  
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
            for idx in valid_idxs:
                row = event_data[idx]
                flow_tensor = self.zero_dem_raster.clone()
                flow_tensor[self.upstream_filters['upstream1']] = float(row[0])
                flow_tensor[self.upstream_filters['upstream2']] = float(row[1])
                flow_tensor[self.upstream_filters['upstream3']] = float(row[2])
                if torch.cuda.is_available():
                    flow_tensor = flow_tensor.cuda()
                dem_input_tensor = self.dem_raster.clone().cuda()
                water_depth_tensor = self.inundation_data_cache[event_id][idx]
                output_tensor = self.inundation_data_cache[event_id][idx+1]
                #flatten the output tesnor to rows * columns
                output_tensor = output_tensor.flatten()
                
                # Combine tensors into a single input tensor
                input_tensor = torch.stack([flow_tensor, dem_input_tensor, water_depth_tensor], dim=0)
                input_data.append(input_tensor)
                output_data.append(output_tensor)  # Add the missing output_tensor argument here
        
        input_batch = torch.stack(input_data, dim=0)
        output_batch = torch.stack(output_data, dim=0)
        # Shuffle the input and output batches together
        # if input_batch.shape[0] > 1:  #Only shuffle if we have more than one sample
        #     indices = torch.randperm(input_batch.shape[0])
        #     input_batch = input_batch[indices]
        #     output_batch = output_batch[indices]
        #     logger.debug(f"Shuffled batch with {len(indices)} samples")
        return input_batch, output_batch
    
    def test_idx_prep(self):
        """Prepare test indices for the test event"""
        if len(self.test_event_ids) == 0:
            logger.error("No test event IDs provided")
            return None
        
        # Create indices for the test event
        start_idx = self.test_start_index
        end_idx = self.test_end_index + 1
        
        # Generate indices for the test event
        test_indices = np.arange(start_idx, end_idx)
        self.test_index = test_indices

    def get_test_batch(self, local_index):
        test_output = []
        test_input_data = []
        test_event_id = self.test_event_ids[0]
        input_arr = self.event_input_map[test_event_id]
        
        local_indices = [local_index]  # Assuming local_index is a single index for the test event
        for local_idx in local_indices:
            
            if local_idx == 0:
                # For the first index, we cannot use the previous timestep so use the first timestep
                flow_tensor = self.zero_dem_raster.clone()
                flow_tensor[self.upstream_filters['upstream1']] = float(input_arr[local_idx, 0])
                flow_tensor[self.upstream_filters['upstream2']] = float(input_arr[local_idx, 1])
                flow_tensor[self.upstream_filters['upstream3']] = float(input_arr[local_idx, 2])
            else:   
                # Get input data
                flow_tensor = self.zero_dem_raster.clone()
                flow_tensor[self.upstream_filters['upstream1']] = float(input_arr[local_idx-1, 0])
                flow_tensor[self.upstream_filters['upstream2']] = float(input_arr[local_idx-1, 1])
                flow_tensor[self.upstream_filters['upstream3']] = float(input_arr[local_idx-1, 2])
            
            if torch.cuda.is_available():
                flow_tensor = flow_tensor.cuda()
            
            dem_input_tensor = self.dem_raster.clone()
            if local_idx == 0:
                # For the first index, we cannot use the previous timestep so use the first timestep
                water_depth_tensor = self.inundation_data_cache[test_event_id][0]
            else:
                water_depth_tensor = self.inundation_data_cache[test_event_id][local_idx-1]
            
            input_tensor = torch.stack([flow_tensor, dem_input_tensor, water_depth_tensor], dim=0).cuda()
            test_input_data.append(input_tensor)
            
            # Get output data (next timestep)
            output_tensor = self.inundation_data_cache[test_event_id][local_idx]
            output_tensor = output_tensor.flatten().cuda()
            test_output.append(output_tensor)
        
        if len(test_input_data) == 0:
            logger.error("No valid test data found")
            return None, None
            
        test_input_batch = torch.stack(test_input_data, dim=0).cuda()
        test_output_batch = torch.stack(test_output, dim=0).cuda()
        
        return test_input_batch, test_output_batch
    
    ## Utility functions  
    def process_inundation_file(self, file_path):
        try:
            with rio.open(file_path) as src:
                data = src.read(1)
                return data.flatten()
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            return None
        
    def find_output_size(self):
        #Load a sample inundation file to determine the output size
        sample_file = f"{self.simulation_data_dir}/Run1-0000.wd"
        sample_data = self.process_inundation_file(sample_file)
        
        #find col and rows of the upstream coordinates
        return sample_data.shape[0]           
    
    def create_filter(self, coordinates):
        sample_file = f"{self.simulation_data_dir}/Run1-0000.wd"
        square_size = 7
        with rio.open(sample_file) as src:
            transform = src.transform
            row, column = coords2rc(transform, (coordinates['easting'], coordinates['northing']))
            square = {
                'row_start': row - square_size // 2,
                'row_end': row + square_size // 2,
                'col_start': column - square_size // 2,
                'col_end': column + square_size // 2
            }
        
        upstream_map = np.zeros_like(self.zero_dem_raster)
        upstream_map[square['row_start']:square['row_end'], square['col_start']:square['col_end']] = 1
        upstream_filter = upstream_map > 0
        return upstream_filter
        
    def find_upstream_point_fiters(self):
        # To create the 2D input from the hydrograph
        upstream_coordinates = [
            {"name": "upstream1", "easting": 342682, "northing": 557532},
            {"name": "upstream2", "easting": 341362, "northing": 554702},
            {"name": "upstream3", "easting": 339947, "northing": 554702},
        ]
        
        for coordinates in upstream_coordinates: 
            # Find the square around the upstream point 7 * 7
            filter = self.create_filter(coordinates)      
            self.upstream_filters[coordinates['name']] =  filter                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    
    def get_no_time_steps(self, event_id):
        inundation_files = glob.glob(f"{self.simulation_data_dir}/Run{event_id}-*.wd")
        return len(inundation_files) - 8 - self.lag
    
    def clear_inundation_cache(self, event_ids=None):
        """Clear cached inundation data for specified events or all events if None"""
        if event_ids is None:
            logger.info("Clearing entire inundation cache")
            self.inundation_data_cache = {}
        else:
            for event_id in event_ids:
                if event_id in self.inundation_data_cache:
                    logger.info(f"Clearing cache for event {event_id}")
                    del self.inundation_data_cache[event_id]
    
    def index_expander(self, event_ids):
        all_indices = []
        if len(event_ids) == 0:
            return np.array([])
        
        # First, collect all valid indices from each event
        for event_id in event_ids:
            start_idx = self.event_start_id_map[event_id]
            num_idx = self.get_no_time_steps(event_id)
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
        
    def idx_prep(self, shuffle=True, random_seed=341):
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
