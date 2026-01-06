import os
import glob
import numpy as np
import torch
import logging

from modules.lib.constants import SIMULATION_DATA_DIR, OUTPUT_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.train_idx = None
        self.validation_idx = None
        self.test_input = None
        self.test_output = None
        self.batch_size = None
        self.test_start_timestep = 1 
        self.test_end_timestep = 266
        self.test_start_index = self.test_start_timestep -1 
        self.test_end_index = self.test_end_timestep - 1
        
    def get_batch(self, indices, subset="train"):
        pass
    
def shuffle_training_data(self, epoch):
    pass
    
def check_inundation_data_cache(event_id):
    cache_dir = os.path.join(OUTPUT_DIR, "preprocessed_inundation")
    if not os.path.exists(cache_dir):
        return False
    cache_file = os.path.join(cache_dir, f"event_{event_id}_inundation.pt")
    if not os.path.exists(cache_file):
        return False
    
    return True

def create_inundation_map_tensors():
    """Preprocess all inundation data and save to disk for fast loading"""
    output_dir = os.path.join(OUTPUT_DIR, "preprocessed_inundation")
    if os.path.exists(output_dir):
        logger.info(f"Preprocessed inundation directory {output_dir} already exists.")
        return
 
    os.makedirs(output_dir, exist_ok=True)
    
    for event_id in range(1, 10):
        if check_inundation_data_cache(event_id):
            logger.info(f"Inundation data for event {event_id} already cached. Skipping preprocessing.")
            continue
        event_inundation_files = glob.glob(f"{SIMULATION_DATA_DIR}/Run{event_id}-*.wd")
        event_inundation_files.sort()
        event_inundation_files = event_inundation_files[8:]  # Skip first 8 filess
        
        if not event_inundation_files:
            logger.warning(f"No inundation files found for event {event_id}")
            return
            
        logger.info(f"Loading {len(event_inundation_files)} inundation files for event {event_id}")
        
        # Load all files for this event
        event_inundation_data = []
        for inundation_file in event_inundation_files:
            inundation_map = gdal_asarray(inundation_file)
            event_inundation_data.append(inundation_map)
        event_inundation_data = np.array(event_inundation_data)
        
        # Save as memory-mappable tensor file
        event_tensor = torch.tensor(np.array(event_inundation_data), dtype=torch.float32)
        output_file = os.path.join(output_dir, f"event_{event_id}_inundation.pt")
        torch.save(event_tensor, output_file)
        logger.info(f"Saved event {event_id}, shape: {event_tensor.shape}")
