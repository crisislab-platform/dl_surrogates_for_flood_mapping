from modules.lib.constants import OUTPUT_DIR, RUN_DIR, CARLISLE_DATA_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib  import gdal_asarray, read_shp_point, coords2rc, gdal_transform, gdal_writetiff
from modules.utils.run_util import check_device
from modules.models.usrr_1dcnn.unet import UNet
from modules.dataloader.raster.raster_loader_unet import UNetDataManager
from modules.models.usrr_1dcnn.unet import model_name as UNET_MODEL_NAME
from modules.model_runner.model_utils import find_model_file
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import glob
import time

pred_ls_file = os.path.join(OUTPUT_DIR, "rls")

logger = logging.getLogger("ReconstructionModule")
logger.setLevel(logging.INFO)

class ReconstructionModule():
    def __init__(self, model_file, sampling_distance, cluster_size, run_dir):
        self.device =  check_device()
        self.load_model_from_file(model_file)
        self.sampling_distance = sampling_distance
        self.cluster_size = cluster_size
        self.run_dir = run_dir
        self.model_file = model_file
        self.dem_asc_file = os.path.join(CARLISLE_DATA_DIR, "Carlisle_5m.asc")
        self.test_event_ids = [1]
        self.simulation_dir  =  SIMULATION_DATA_DIR


    def load_reference_data(self):
        rl_file_path = os.path.join(OUTPUT_DIR,"rls", "ss_20.shp")
        self.rep_locations= read_shp_point(rl_file_path)
        self.rl_group_size = len(self.rep_locations)
        transform  = gdal_transform(self.dem_asc_file)
        self.dem_map = gdal_asarray(self.dem_asc_file)
        self.rl_rcs = [coords2rc(transform, coord)  for coord in self.rep_locations]
        self.rl_map = np.zeros(self.dem_map.shape)
        for row, col in self.rl_rcs:
            self.rl_map[row, col] = 1
       
        filter_mask  = self.rl_map == 1
        self.rl_filter = lambda x: x[filter_mask]
        
        inundation_files = glob.glob(f"{self.simulation_dir}/Run{self.test_event_ids[0]}-*.wd")
        inundation_files.sort()
        inundation_files = inundation_files[8:]  # Skip the first 8 rows
        inundation_data_arr = []
    
        for t_idx, inundation_file in enumerate(inundation_files):
            inundation_data = gdal_asarray(inundation_file)
            timestep_rl_inundation = self.rl_filter(inundation_data)
            inundation_data_arr.append(timestep_rl_inundation)
        
        inundation_data_arr = np.array(inundation_data_arr)
        logger.info(f"Shape of inundation data array: {inundation_data_arr.shape}")
        return inundation_data_arr
        
    def depth_at_rls(self, sampling_distance, cluster_size):
        # file_path  = os.path.join(pred_ls_file, f"rls_depth_{sampling_distance}_{cluster_size}.tif")
        # depths = gdal_asarray(file_path)
        # Load the RL file
        
        reference_depth_data = self.load_reference_data()
        prediction_depth_data = reference_depth_data
        # Add some dummy data
        
        return prediction_depth_data, reference_depth_data
    
    
    def save_depth_map(self, depth_map, event_id, map_i):
        depth_map_file = os.path.join(self.run_dir, f"depth_map_event_{event_id}_map_{map_i}.tif")
        gdal_writetiff(depth_map, depth_map_file, self.dem_asc_file)
        logger.info(f"Depth map saved to {depth_map_file}")
       
    def reconstruct(self):
        loss_fn = nn.MSELoss()
        pred_depths, ref_depth = self.depth_at_rls(self.sampling_distance, self.cluster_size)
        reco_model = UNetDataManager(sampling_dist=self.sampling_distance, run_dir=self.run_dir)
        reco_model.reconstruction_init()
        
        num_of_test_events = len(reco_model.test_event_ids)
        test_idxs = reco_model.test_idxs.reshape((num_of_test_events,-1, reco_model.num_of_batch_per_map))
        ext_mask = gdal_asarray(reco_model.possible_inun_file)
        ext_mask = ~np.isnan(ext_mask)
        
        losses = []
        start_time = time.time()
        for event_i in range(num_of_test_events):
            for map_i in range(test_idxs.shape[1]): 
                pred_depths_map = pred_depths[map_i, :]
                depth_map, ref_map = self.single_construct(reco_model, self.model, pred_depths_map, 
                                                            test_idxs[event_i, map_i], ext_mask, onlyspecs = False)
                logger.info(f"Depth map shape: {depth_map.shape}")
                loss = loss_fn(depth_map, ref_map)
                losses.append(loss)
                logger.info(f"Loss {loss}")
                if map_i == 145:
                    depth_map = depth_map.detach().cpu().numpy()
                    self.save_depth_map(depth_map, self.test_event_ids[event_i], map_i)
        end_time  = time.time()
        
        #Calcuate metrics
        mse = np.mean(losses)
        rmse = np.sqrt(mse)
        reconstruction_time = end_time - start_time
        return mse, rmse, reconstruction_time
                               
    def single_construct(self, reco_model:UNetDataManager, model, pred_depths, test_idxs, ext_mask, onlyspecs = False):
        with torch.no_grad():
            # Do prediction for one batch
            input_batch, ref_map = reco_model.get_batch(test_idxs[0], test_mode=True, rl_depth=pred_depths)
            preds = model(input_batch).detach().cpu()
            
            for batch_i in test_idxs[1:]:
                input_batch, _ = reco_model.get_batch(batch_i, test_mode=True, rl_depth=pred_depths)
                preds = torch.cat((preds, model(input_batch).detach().cpu()), dim=0)
            del _
            torch.cuda.empty_cache()
            depth_map = reco_model.reconstruct_full_map(preds.to(reco_model.device))
            logger.info(f"Depth map shape: {depth_map.shape}")
            torch.cuda.empty_cache()
            ref_map[ref_map <0.05] = 0
            depth_map[depth_map <0.05] = 0
            return depth_map, ref_map
            
    def load_model_from_file(self, model_file):
        checkpoint  = torch.load(model_file)
        model_structue  = checkpoint['model_structure']
        self.model = UNet(enc_chs= model_structue, dec_chs=model_structue[:0:-1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()   
        return 0
    
def validate_reconstruction(model_run_id, sampling_distance, cluster_size):
    model_file  = find_model_file(UNET_MODEL_NAME, model_run_id)
    if model_file is None:
        raise ValueError(f"Model file not found for sampling distance {sampling_distance}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUN_DIR, "reconstruction", run_id)
    os.makedirs(run_dir, exist_ok=True)
    reco_model = ReconstructionModule(model_file, sampling_distance, cluster_size, run_dir)
    reco_model.reconstruct()







