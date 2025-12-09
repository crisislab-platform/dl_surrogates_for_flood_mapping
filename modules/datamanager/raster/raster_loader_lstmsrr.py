# Description: Data loader for U-Net model (PyTorch version)
from modules.lib.constants import DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib import coords2rc, rc2coords, gdal_asarray, gdal_transform, gdal_writetiff
from modules.models.usrr_1dcnn.lib.base_functions import read_shp_point
from modules.utils.run_util import check_device
from modules.datamanager.datamanager import check_inundation_data_cache

import numpy as np
import glob
import logging
import torch
from modules.datamanager.datamanager import DataManager
import os

logger = logging.getLogger("LSRM_SRR_DataManager")
logger.setLevel(logging.INFO)

class ReconsturctionDataManager(DataManager):
    
    def __init__(self):
        super().__init__()
    
        self.device = check_device()
        self.test_event_ids = [1]
        self.preloaded_maps = []
        
        self.preload_inundation_maps()
        self.prepare_reconstruction_test_idxs()
        logger.info("Reconstruction mode initialized")
        return
        
    def prepare_reconstruction_test_idxs(self):        
        self.reconstruction_test_idxs = np.arange(len(self.preloaded_maps))
        logger.info(f"Prepared {len(self.reconstruction_test_idxs)} timesteps for reconstruction testing")

    def get_batch(self, t_idx):
        with torch.no_grad():
            inundation_map = self.preloaded_maps[t_idx]
            return inundation_map
            
    def preload_inundation_maps(self):
        with torch.no_grad():
            if check_inundation_data_cache(self.test_event_ids[0]):
                logger.info("Preloading inundation maps")
                self.preloaded_maps = []
                inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{self.test_event_ids[0]}_inundation.pt"))
                for i in range(len(inundation_data)):
                    tensor = inundation_data[i].cuda() 
                    self.preloaded_maps.append(tensor)
                logger.info(f"Preloaded and processed inundation maps for reconstruction: {len(self.preloaded_maps)} timesteps")   
            else:
                raise ValueError("Inundation data cache not found. Please run create_inundation_map_tensors() first.")