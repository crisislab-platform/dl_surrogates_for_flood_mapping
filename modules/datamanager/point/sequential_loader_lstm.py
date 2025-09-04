import os
import pandas as pd
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR, RUN_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray, read_shp_point,gdal_transform
from modules.models.srr_lstm.srr.gdal_func import coords2rc, ogr, gdal
import torch
import logging

from modules.datamanager.datamanager import DataManager, check_inundation_data_cache
from modules.datamanager.raster.raster_loader_lstmsrr import ReconsturctionDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTMSRRDataManager")

class LSTMSequentialDataManager(DataManager):
    
    def __init__(self, batch_size=32, input_time_len_h=1, rl_id=1, fold=1, tuning_mode=True, reconstruction_mode=False, reco_data_manager:ReconsturctionDataManager=None):
        super().__init__()
        
        # Directories
        self.simulation_data_dir = SIMULATION_DATA_DIR
        self.dem_file = os.path.join(self.simulation_data_dir, "Carlisle_5m.asc")
        self.dem_dataset = gdal.Open(self.dem_file)
        self.reconstruction_mode = reconstruction_mode
        self.tuning_mode = tuning_mode
        self.rep_locattion_filepath = os.path.join(OUTPUT_DIR, "sdr_reduction_results" , "representative_locations.shp")
        
        # Prepare train, test, and validation event idsx
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
        
        # Representaitve location variables
        self.rep_location_id = int(rl_id)
        self.rep_localtion = (0,0) #row, col
        self.rep_localtion_coords = None
        
        # Prepare the inputs and outputs
        self.event_batch_map = {}
        self.inundation_data_cache = {}
        self.find_rep_location_and_prepare_filter()

        if not self.reconstruction_mode:
            self.preload_inundation_data() 
        else:
            self.inundation_data_cache[self.test_event_ids[0]] = self.reco_data_manager.preloaded_maps
            self.inundation_data_cache[self.test_event_ids[0]] = [self.rl_filter(tensor) for tensor in self.inundation_data_cache[self.test_event_ids[0]]]
        
        self.input_tensor_prep()
        
        if not self.reconstruction_mode:
            self.prepare_batch_idxs()
        
        if not self.tuning_mode:
            test_sequences = self.test_sequences
            test_input = []
            test_output = []
            for input_tensor, output_tensor in test_sequences:
                test_input.append(input_tensor)
                test_output.append(output_tensor)
            test_input = torch.stack(test_input, axis=0)
            test_output = torch.stack(test_output, axis=0)
            self.test_input = test_input.cuda()
            self.test_output = test_output.cuda()
            
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
            return idx_batch_list
        else:
            logger.error("No batches were created! Check your data and batch size.")
            return np.array([]) 
        
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
    
    def input_tensor_prep(self):
        # First load and normalize raw inflow data for each event
        raw_inflow_data = {}
        all_train_data = []
        
        # 1. Load all raw data
        for event_id in self.all_event_ids:
            inflow_file = os.path.join(CARLISLE_DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
            inflow_data = pd.read_csv(inflow_file)
            
            inflow_data = inflow_data.iloc[8:,:] # Skip the first 8 rows
            inflow_data = inflow_data.values
            inflow_data = inflow_data[:, 1:] # Skip the first column
            input_arr = np.array(inflow_data.astype(np.float32))
            
            # Add padding for the first few input sequences
            input_arr = np.r_['0,2', np.zeros((self.input_seq_length-1, input_arr.shape[1])), input_arr]
            input_arr[:self.input_seq_length-1, :] = np.repeat(input_arr[self.input_seq_length-1:self.input_seq_length, :], self.input_seq_length-1, axis=0) 
            raw_inflow_data[event_id] = input_arr
            
            # Only collect training data for fitting the scaler
            if event_id in self.train_event_ids:
                all_train_data.append(input_arr)
        
        ## There is an scaler issue, fix it later. Need the scaler fited from all the training data. 
        # 2. Normalize the data using MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        all_train_data = np.vstack(all_train_data)
        scaler.fit(all_train_data)
        
        self.train_sequences = []
        for event_id, raw_data in raw_inflow_data.items():
            # Normalize by max inflow
            normalized_data = scaler.transform(raw_data)
            n_samples = len(normalized_data) - self.input_seq_length + 1
            
            # Input_sequences = np.zeros((n_samples, self.input_seq_length, n_features))
            input_sequences = [normalized_data[i: i + self.input_seq_length, :] for i in range(n_samples)]
            input_sequences = np.array(list(input_sequences)) 
            
            outputs = self.inundation_data_cache[event_id]
            input_sequences = torch.from_numpy(input_sequences).float().cuda()
            outputs = torch.stack(outputs, dim=0).cuda()
            
            if outputs.shape[0] != input_sequences.shape[0] and event_id not in self.test_event_ids:
                logger.error(f"Mismatch in output length for event {event_id}. Expected: {n_samples}, Found: {len(outputs)}")
                raise ValueError(f"Mismatch in output length for event {event_id}. Expected: {n_samples}, Found: {len(outputs)}")
           
            # Create a pair of input and output sequences ((input_tensor, output_tensor), ...)
            sequence_pairs = []
            for i in range(input_sequences.shape[0]):
                sample_input = input_sequences[i] 
                sample_output = outputs[i] # Reshape to match the expected output shape.reshape(1, 1) 
                sequence_pairs.append((sample_input, sample_output))

            if event_id in self.train_event_ids:
                self.train_sequences.extend(sequence_pairs)
            if event_id in self.test_event_ids:
                self.test_sequences = sequence_pairs
            if event_id in self.val_event_ids:
                self.val_sequences = sequence_pairs
                
        # Concatenate all training sequences into a single array
        logger.info("Completed normalization and sequence creation")

    def preload_inundation_data(self):
        for event_id in self.all_event_ids:
            if check_inundation_data_cache(event_id):
                inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{event_id}_inundation.pt"))
                self.inundation_data_cache[event_id] = []
                for i in range(len(inundation_data)):
                    tensor = inundation_data[i].cuda() 
                    filtered_tensor = self.rl_filter(tensor)
                    self.inundation_data_cache[event_id].append(filtered_tensor)
                logger.info(f"Inundation cache size:  {inundation_data.shape}")
                logger.info(f"Loaded inundation data for event {event_id} from cache")
            else:
                raise ValueError(f"Inundation data for event {event_id} not found in cache. Please run create_inundation_map_tensors() first.")
 
    def find_rep_location_and_prepare_filter(self):
        """Find the representative location using the shapefile and representative location id"""

        # Read the DEM file for reference
        self.dem_map = gdal_asarray(self.dem_file)
        
        # Use geopandas to read the shapefile with attributes
        import geopandas as gpd
        gdf = gpd.read_file(self.rep_locattion_filepath)
        
        # Find the point with the matching ID
        matching_point = gdf[gdf["PointID"] == self.rep_location_id]
        
        if matching_point.empty:
            logger.error(f"No point with ID {self.rep_location_id} found in {self.rep_locattion_filepath}")
            exit(1)
        
        # Extract coordinates from the matching point
        point_geometry = matching_point.iloc[0].geometry
        x_coord = matching_point.iloc[0]["X_Coord"]
        y_coord = matching_point.iloc[0]["Y_Coord"]
        self.rep_localtion_coords = (x_coord, y_coord)
        
        logger.info(f"Found representative location with ID {self.rep_location_id} at coordinates: {self.rep_localtion_coords}")
        
        # Convert coordinates to row, col in the DEM grid

        self.dem_transform = self.dem_dataset.GetGeoTransform()
        row, col = coords2rc(self.dem_transform,(x_coord, y_coord))
        self.rep_localtion = (row, col)
        logger.info(f"Representative location at row={row}, col={col}")
        
        # Prepare the filter mask for representative locations
        rl_mask = np.zeros_like(self.dem_map, dtype=np.float32)
        rl_mask[self.rep_localtion[0], self.rep_localtion[1]] = 1
        self.filter_mask  = rl_mask == 1
        self.rl_filter = lambda x: x[self.filter_mask]
        # self.visualise_rep_location()

    def visualise_rep_location(self):
        """Get the representative locations using the rep_location_filepath and visualize them on DEM"""
        import matplotlib.pyplot as plt
        
        # Create a visualization
        plt.figure(figsize=(12, 10))
        
        # Plot the DEM as background
        plt.imshow(self.dem_map, cmap='terrain', alpha=0.7)
        plt.colorbar(label='Elevation')
        
        # Plot all representative locations
        rows_cols = [(self.rep_localtion[0], self.rep_localtion[1])]
        rows, cols = zip(*rows_cols) if rows_cols else ([], [])
        plt.scatter(cols, rows, c='blue', marker='x', label=f'Representaitve location {self.rep_location_id}', s=30, alpha=0.6)
        
        plt.title(f'Representative Locations (Cluster {self.rep_location_id})')
        plt.legend()
        
        # Save the visualization
        output_path = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"rls_{self.rep_location_id}_plot.png"))
        plt.close()
        
        logger.info(f"Visualization of representative locations saved to {output_path}")
        