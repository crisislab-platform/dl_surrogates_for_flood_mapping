import os
import glob
import numpy as np
import torch
import logging
import pandas as pd

from modules.lib.constants import SIMULATION_DATA_DIR, OUTPUT_DIR, DATA_DIR, BC_DATA_DIR, STUDY_AREA, FLOOD_MAPS_DIR, DEM_FILE
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray
from modules.utils.run_util import check_device
import os
import rasterio as rio
from modules.datamanager.spatial_sampler.flood_map_sampler import FloodMapSampler

logger = logging.getLogger(__name__)

class DataSet(torch.utils.data.Dataset):
    
    def __init__(self, data_manager, subset="train"):
        self.data_manager = data_manager
        self.subset = subset
        if subset == "train":
            self.indices = self.data_manager.train_idx
        elif subset == "validation":
            self.indices = self.data_manager.validation_idx
        elif subset == "test":
            self.indices = self.data_manager.test_idx
        else:
            raise ValueError(f"Invalid subset: {subset}. Must be 'train', 'validation', or 'test'.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data_manager.get_batch(self.indices[idx], subset=self.subset)

class DataManager:
    
    def __init__(self, config):
        self.config = config
        self.train_event_ids = config.train_events 
        self.validation_event_ids = config.validation_events
        self.test_event_ids = config.test_events 
        self.batch_size = config.batch_size
        self.tuning_mode = config.tuning_mode
        
        self.indices_per_timestep = config.indices_per_timestep
        self.all_event_ids = np.concatenate([self.train_event_ids, self.test_event_ids, self.validation_event_ids])
        self.all_event_ids = np.unique(self.all_event_ids)
        self.all_event_ids.sort()
        
        self.event_index_map, self.event_name_map = self.create_indices()
        self.device = check_device()
        
        self.train_idx = None
        self.validation_idx = None
        self.test_idx = None
        self.sample_flood_map = None
        self.train_data_loader = None
        self.validation_data_loader = None
        self.test_data_loader = None
        self.flood_map_file_map = {}
        self.run_dir = config.run_dir
        
        # Spatial sampling parameters
        self.sampling_dist = config.args.get('sampling_distance', 265)  # Default to 512 if not provided
        self.tile_resolution = config.args.get('tile_resolution', 512)  # Default to 512 if not provided
        
        if self.indices_per_timestep > 1:
            self.output_shape = self.find_study_domain_shape()
            self.sample_flood_map = self.find_sample_flood_map()
            self.map_sampler = FloodMapSampler(self.sampling_dist, self.run_dir, self.tile_resolution,  self.sample_flood_map, self.output_shape)
            self.tiles_per_index, self.number_of_tiles_per_timestep = self.map_sampler.calculate_batch_size_and_number_of_tiles(self.indices_per_timestep)  

    
    def create_indices(self):
        if STUDY_AREA == "carlisle":
            event_index = {
                1: "Run1",
                2: "Run2",   
                3: "Run3",
                4: "Run4",
                5: "Run5",
                6: "Run6",
                7: "Run7",
                8: "Run8",
                9: "Run9",  
            }
            event_name_map = {
                "Run1": "Run1",
                "Run2": "Run2",
                "Run3": "Run3",
                "Run4": "Run4",
                "Run5": "Run5",
                "Run6": "Run6",
                "Run7": "Run7",
                "Run8": "Run8",
                "Run9": "Run9"
            }
            
        elif STUDY_AREA == "westport":
            event_index = {
                1: "Base0",
                2: "Base1",
                3: "Base2",
                4: "Base3",
                5: "Base4",
                6: "Base5",
                7: "Base6",
                8: "Base7",
                9: "Base8", 
            }
            for i in range(1,98):
                event_index[9+i] = f"S{i:02d}"
                
            event_name_map = {
                "Base0": "Base0_Buller20yrARI_v2",
                "Base1": "Base1_Buller50yrARI_v2",
                "Base2": "Base2_Buller100yrARI",
                "Base3": "Base3_Buller50yrARI_RCP4pt5",
                "Base4": "Base4_Buller50yrARI_RCP6pt0",
                "Base5": "Base5_Buller50yrARI_RCP8pt5",
                "Base6": "Base6_Buller100yrARI_RCP4pt5",
                "Base7": "Base7_Buller100yrARI_RCP6pt0",
                "Base8": "Base8_Buller100yrARI_RCP8pt5"
            }
            
            for i in range(1,98):
                event_name_map[f"S{i:02d}"] = f"S{i:02d}"
        return event_index, event_name_map
        
    def get_batch(self, indices, subset="train"):
        pass

    def create_dataloaders(self, num_workers=0):
        # Create datasets
        logger.info("Creating datasets for training, validation, and testing")
        train_dataset = DataSet(self, subset="train")
        validation_dataset = DataSet(self, subset="validation")
        test_dataset = DataSet(self, subset="test")

        #Use num_workers=0 for GPU training to avoid memory conflicts from parallel data loading
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        logger.info("Data loaders created successfully")
    
    def load_inundation_data(self, event_id, files):
        event_name = self.event_name_map.get(self.event_index_map[event_id], None)
        if event_name is None:
            logger.warning(f"Unknown event ID: {event_id}. Cannot load inundation data.")
            return []
        
        inundation_tensors = []
        for file in files:
            if STUDY_AREA == "westport":
                file_path = os.path.join(FLOOD_MAPS_DIR, f"{event_name}", file)
            elif STUDY_AREA == "carlisle":
                file_path = os.path.join(FLOOD_MAPS_DIR, f"{event_name}", file)
            inundation_data = self.process_inundation_file(file_path)
            if inundation_data is not None:
                tensor_data = torch.from_numpy(inundation_data).float().to(self.device)
                inundation_tensors.append(tensor_data)
        return inundation_tensors
    
    def shuffle_training_data(self, epoch):
        pass
    
    def find_study_domain_shape(self):
        if self.sample_flood_map is not None:
            sample_data = gdal_asarray(self.sample_flood_map)
            return sample_data.shape
        
        # Load a sample inundation file to determine the spatial dimensions of the study domain
        test_event_name = self.event_name_map.get(self.event_index_map[self.train_event_ids[0]], None)
        if STUDY_AREA == "carlisle":
            inundation_files = glob.glob(os.path.join(FLOOD_MAPS_DIR, test_event_name, "Run*.wd"))
        elif STUDY_AREA == "westport":
            inundation_files = glob.glob(os.path.join(FLOOD_MAPS_DIR, f"{test_event_name}_depth_snapshots", "depth_at_*.tif"))
        inundation_files.sort()
        sample_file = inundation_files[0] if inundation_files else None
        if sample_file is None:
            raise ValueError(f"No inundation files found for event {self.train_event_ids[0]}")
        sample_data = gdal_asarray(sample_file)
        self.sample_flood_map = sample_file  # Cache the sample flood map path for later use
        return sample_data.shape # Return height and width
    
    def find_sample_flood_map(self):
        if hasattr(self, 'sample_flood_map') and self.sample_flood_map is not None:
            return self.sample_flood_map
        
        # Load a sample inundation file to use as a template for tiling and padding
        test_event_name = self.event_name_map.get(self.event_index_map[self.test_event_ids[0]], None)
        if STUDY_AREA == "carlisle":
            inundation_files = glob.glob(f"{FLOOD_MAPS_DIR}/f{test_event_name}/Run*.wd")
        elif STUDY_AREA == "westport":
            inundation_files = glob.glob(f"{FLOOD_MAPS_DIR}/{test_event_name}_depth_snapshots/depth_at_*.tif")
        inundation_files.sort()
        sample_file = inundation_files[0] if inundation_files else None
        if sample_file is None:
            raise ValueError(f"No inundation files found for event {self.test_event_ids[0]}")
        self.sample_flood_map = sample_file  # Cache the sample flood map path for later use
        return sample_file
        
    def extract_timestep_from_filename(self, filename):
        try:
            if STUDY_AREA == "westport":
                timestep_str = filename.split("depth_at_")[1].split(".tif")[0]
                timestep_dt = pd.to_datetime(timestep_str, format="%Y-%m-%d_%H-%M-%S")
            elif STUDY_AREA == "carlisle":
                timestep_str = filename.split("-")[1].split(".wd")[0]
                timestep_dt = int(timestep_str)
            return timestep_dt
        except Exception as e:
            logger.error(f"Error extracting timestep from filename {filename}: {e}")
            return None
        
    def process_inundation_file(self, file_path):
        return process_inundation_file(file_path)
    
    def get_no_time_steps(self, event_id):
        return len(self.flood_map_file_map[event_id])
    
        
    def prepare_batch_indices(self):
        self.event_start_id_map = {}
        # Calculate start indices for each event
        for event_id in self.all_event_ids:
            if event_id == 1:  # First event
                start_idx = 0
            else:
                prev_event_id = event_id - 1
                start_idx = self.event_start_id_map[prev_event_id] + self.get_no_time_steps(prev_event_id) * self.indices_per_timestep
            self.event_start_id_map[event_id] = start_idx
  
        self.train_idx = self.index_expander(self.train_event_ids)
        self.validation_idx = self.index_expander(self.validation_event_ids)
        self.test_idx = self.index_expander(self.test_event_ids)
        
    def index_expander(self, event_ids):
        all_indices = []
        if len(event_ids) == 0:
            return np.array([])

        # First, collect all valid indices from each event
        for event_id in event_ids:
            start_idx = self.event_start_id_map[event_id]
            num_idx =  self.get_no_time_steps(event_id) * self.indices_per_timestep
            event_indices = start_idx + np.arange(num_idx)
            all_indices.extend(event_indices)
        
        all_indices = np.array(all_indices)
        if all_indices.shape[0] > 0:
            logger.info(f"Index batches shape: {all_indices.shape}")
            logger.info(f"Created mixed-event batches with data from multiple events")
            return all_indices
        else:
            logger.error("No batches were created! Check your data and batch size.")
            return np.array([])  # Return empty array as fallback
        
    def find_local_indices(self, event_id, idx):  
        start_idx = self.event_start_id_map[event_id]
        local_idx = idx - start_idx
        timestep_idx = local_idx // self.indices_per_timestep
        # Calculate which tile group within the map this index belongs to
        tile_group = local_idx % self.indices_per_timestep
        
        # Make sure the indices are valid for this event
        maps_in_event = self.get_no_time_steps(event_id)
        if 0 <= timestep_idx < maps_in_event:
            return (timestep_idx, tile_group)
        logger.warning(f"Index {idx} is out of bounds for event {event_id} with {maps_in_event} maps")
        return None, None
    


    def find_event_id(self, idx):
        for event_id in self.all_event_ids:
            start_idx = self.event_start_id_map[event_id]
            end_idx = start_idx + self.get_no_time_steps(event_id) * self.indices_per_timestep - 1
            if start_idx <= idx <= end_idx:
                return event_id
        logger.warning(f"Could not find event ID for index {idx}")
        return None
    
        

    def get_flood_map(self, event_id, time_step):
        flood_map_file = self.flood_map_file_map[event_id][time_step]
        if flood_map_file is None:
            logger.warning(f"No flood map file found for event {event_id} at time index {time_step}")
            return None
        flood_map_tensor = self.load_inundation_data(event_id, [flood_map_file])[0]
        return flood_map_tensor
    
 
def process_inundation_file(file_path):
    try:
        with rio.open(file_path) as src:
            data = src.read(1)
            nodata_value = src.nodata
            data = np.where(np.isnan(data), 0, data)
            data = np.where(data == nodata_value, 0, data)
            
            return data
    except Exception as e:
        logger.error(f"Error processing inundation file {file_path}: {e}")
        return None  
    