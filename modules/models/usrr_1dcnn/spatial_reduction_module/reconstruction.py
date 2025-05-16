from modules.lib.constants import OUTPUT_DIR, RUN_DIR, CARLISLE_DATA_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib  import gdal_asarray, read_shp_point, coords2rc, gdal_transform, gdal_writetiff
from modules.utils.run_util import check_device
from modules.models.usrr_1dcnn.unet import UNet
from modules.models.usrr_1dcnn.cnn1d import CNN1DSequential
from modules.datamanager.raster.raster_loader_unet import UNetDataManager
from modules.datamanager.point.sequential_loader_1dcnn import CNNSequentialDataManager
from modules.models.usrr_1dcnn.unet import model_name as UNET_MODEL_NAME
from modules.lib.constants import USRR_1DCNN_V1
from modules.model_runner.model_utils import find_model_file
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import logging
import glob
import time
import rasterio
import concurrent.futures

pred_ls_file = os.path.join(OUTPUT_DIR, "rls")

logger = logging.getLogger("ReconstructionModule")
logger.setLevel(logging.INFO)

class ReconstructionModule():
    
    def __init__(self, uent_model_file, cnn_model_files, sampling_distance, cluster_size, run_dir):
        self.device =  check_device()
        self.input_time_len_h = 10
        self.seq_h = self.input_time_len_h * 4
        self.sampling_distance = sampling_distance
        self.cluster_size = cluster_size
        self.time_lag_h = 0
        self.timestep =1 
        self.run_dir = run_dir
        self.model_file = uent_model_file
        self.dem_asc_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
        self.simulation_dir  =  SIMULATION_DATA_DIR
        self.rl_file_path = os.path.join(OUTPUT_DIR, "rls", f"ss_{self.sampling_distance}.csv")
        self.rep_loc_file_path = f"{OUTPUT_DIR}/rls/ss_{sampling_distance}.csv"
        self.unet = self.load_model_from_file(uent_model_file)
        self.cnn_models = self.load_cnn_models(cnn_model_files)
        self.test_event_ids = [1]
    
    def load_reference_data(self):
        coords_df = pd.read_csv(self.rl_file_path)
        coordinates = coords_df[['x', 'y']].values
        self.rep_locations= np.array(coordinates)
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
        inundation_files = inundation_files[:190]
        inundation_data_arr = []
    
        for t_idx, inundation_file in enumerate(inundation_files):
            inundation_data = gdal_asarray(inundation_file)
            timestep_rl_inundation = self.rl_filter(inundation_data)
            inundation_data_arr.append(timestep_rl_inundation)
            
        inundation_data_arr = np.array(inundation_data_arr)
        inundation_data_arr = inundation_data_arr.reshape(-1, len(self.rl_rcs))
        
        logger.info(f"Shape of inundation data array: {inundation_data_arr.shape}")
        return inundation_data_arr
    
    def find_representative_locations(self):
        if self.rep_loc_file_path and os.path.exists(self.rep_loc_file_path):
                # Load representative locations from file if provided
                rl_list = pd.read_csv(self.rep_loc_file_path)
                logger.info(f"rep loc: {rl_list.iloc[0]}")
                rl_list = rl_list[['x', 'y']].values
                rl_list = [tuple(x) for x in rl_list]
                logger.info(f"Loaded {len(rl_list)} representative locations from {self.rep_loc_file_path}")
        else:
            raise ValueError(f"RL file path is not valid: {self.rep_loc_file_path}")
        
        self.rl_list = rl_list
        transform  = gdal_transform(self.dem_asc_file)
        
        # Convert RL coordinates to row/col indices
        self.rc_rl_points = [coords2rc(transform, rl) for rl in self.rl_list] 
        length_of_rl_points = len(self.rc_rl_points)
        logger.info(f"Number of representative locations: {length_of_rl_points}")
        
        # Create a binary map with representative locations. 1 at RL, 0 elsewhere
        rl_map = np.zeros(self.dem_map.shape)
        for row, col in self.rc_rl_points:
            rl_map[row, col] = 1
        self.rl_map = rl_map
        
    
        
    def predict_rl_depth(self):
        reference_depth_data = self.load_reference_data()
        self.find_representative_locations()
        all_preds = []
        
        pred_start = time.time()
        # Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all prediction tasks
            future_to_group = {
                executor.submit(self.get_rl_group_predictions, rl_group, model): rl_group
                for rl_group, model in self.cnn_models.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_group):
                rl_group = future_to_group[future]
                try:
                    y_hat_np, data_manager = future.result()
                    rl_locations = data_manager.coords_to_cluster_rls # [(1,2), (3,4)),..]
                    #get the first element of the tuple for each rl location in rl_locations
                    row_indices = [rl[0] for rl in rl_locations]
                    col_indices = [rl[1] for rl in rl_locations]
                    # Process each timestep
                    for i in range(y_hat_np.shape[0]):
                        # Initialize map for this timestep if needed
                        if len(all_preds) <= i:
                            all_preds.append(np.zeros(self.dem_map.shape))
                        
                        # Add predictions to the map at the proper locations
                        all_preds[i][row_indices, col_indices] = y_hat_np[i]
                    
                    logger.info(f"Completed predictions for RL group {rl_group}")
                except Exception as exc:
                    logger.error(f"RL group {rl_group} generated an exception: {exc}")
        pred_end = time.time()
        pred_time = pred_end - pred_start
        logger.info(f"Total prediction time: {pred_time:.4f} seconds")
        logger.info(f"Completed predictions for {len(all_preds)} RL groups in parallel")
        return np.array(all_preds), reference_depth_data, pred_time
        
    def get_rl_group_predictions(self, rl_group, model):
        total_prediction_time = 0  # Initialize variable used in the method
        cnn_data_manager = CNNSequentialDataManager(None, self.input_time_len_h, self.time_lag_h, 
                                             rl_group, self.timestep, self.sampling_distance, 
                                             self.cluster_size, test_mode=True)
        with torch.no_grad():
            device = check_device()
            model.eval()
            test_inputs = cnn_data_manager.test_inputs[0]
            test_inputs = np.array(test_inputs)
            test_inputs = torch.from_numpy(test_inputs)
            model = model.to(self.device)
            test_inputs = test_inputs.to(device)
            start_time = time.time()    
            y_hat = model(test_inputs.float())
            end_time = time.time()
            prediction_time = end_time - start_time
            total_prediction_time += prediction_time
            logger.info(f"Prediction time for {rl_group}: {prediction_time:.4f} seconds")
            y_hat_np = y_hat.cpu().numpy()
            return y_hat_np, cnn_data_manager
                    
    def save_depth_map(self, depth_map, map_i):
        depth_map = depth_map.detach().cpu().numpy()
        depth_map_file = os.path.join(self.run_dir, f"map_{map_i:04d}.wd")
       
        logger.info(f"Saving depth map saved to {depth_map_file}")
        ref_file = os.path.join(SIMULATION_DATA_DIR, "Run1-0000.wd")
        
        # Get dimensisons from reference file
        with rasterio.open(ref_file) as src:
            height = src.height
            width = src.width
            profile = src.profile
            transform = src.transform
            crs = src.crs
        
        logger.info(f'Prediction shape: {depth_map.shape}, Reshaping to dimensions: {height}x{width}')
        pred_reshaped = depth_map.reshape(height, width)
    
        # Create a copy of the profile for the output file
        out_profile = profile.copy()
        out_profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw'
        )
    
        # Write the reshaped prediction directly to a .wd file
        with rasterio.open(depth_map_file, 'w', **out_profile) as dst:
            dst.write(pred_reshaped.astype(rasterio.float32), 1)
        
        logger.info(f"Saved prediction map {map_i} to {depth_map_file}")
        return 0
       
    def reconstruct(self):
        
        loss_fn = nn.MSELoss()
        reco_model = UNetDataManager(sampling_dist=self.sampling_distance, run_dir=self.run_dir)
        reco_model.reconstruction_init()
        num_of_test_events = len(reco_model.test_event_ids)
        test_idxs = reco_model.test_idxs.reshape((num_of_test_events,-1, reco_model.num_of_batch_per_map))
        ext_mask = gdal_asarray(reco_model.possible_inun_file)
        ext_mask = (~np.isnan(ext_mask))
    
        losses = []
        batch_nse = []
        start_time = time.time()
        logger.info("Running reconstruction")
        logger.info(f"Test indices : {test_idxs.shape}")

        logger.info("Predicting depths at RLS")
        pred_depths, ref_depth, pred_time = self.predict_rl_depth()
        logger.info("Time taken for RL depth prediction: {pred_time}")
        
        for event_i in range(num_of_test_events):
            
            for map_i in range(test_idxs.shape[1]):
                logger.info(f"Processing event {event_i}, map {map_i}")
                # Get the predicted depths for the current map               
                pred_depths_map = pred_depths[map_i]
                depth_map, ref_map = self.single_construct(reco_model, 
                                                            pred_depths_map, 
                                                            test_idxs[event_i, map_i], ext_mask, onlyspecs = False)
                logger.info(f"Depth map shape: {depth_map.shape}")
                logger.info(f"Ref map shape: {ref_map.shape}")       
                # Ensure both tensors are on the same device
                depth_map = depth_map.to(self.device)
                ref_map = ref_map.to(self.device)
                loss = loss_fn(depth_map, ref_map)
                nse = self.nse_fn(depth_map, ref_map)
                batch_nse.append(nse)
                losses.append(loss.item())  # Convert tensor to scalar
                logger.info(f"Loss {loss.item()}")
                if map_i == 145:
                    self.save_depth_map(depth_map, map_i)
                    
        end_time = time.time()
        mse = np.mean(losses)
        rmse = np.sqrt(mse)
        nse = np.mean(batch_nse)
        reconstruction_time = end_time - start_time
        
        metrics  = {
            'mse': mse,
            'rmse': rmse,
            'nse': nse,
            'pred_time': reconstruction_time,
            'wet_rmse': None,
            'wet_acc': None,
            'flops': None,
        }
        return metrics
    
    def nse_fn(self, observed, predicted):
        # Make sure both arguments are on the same device
        if observed.device != predicted.device:
            predicted = predicted.to(observed.device)
            
        observed_mean = torch.mean(observed)
        numerator = torch.sum((observed - predicted) ** 2)
        denominator = torch.sum((observed - observed_mean) ** 2)
        nse = 1 - (numerator / denominator)
        nse = nse.item() 
        return nse
                               
    def single_construct(self, reco_model:UNetDataManager, pred_depths, test_idxs, ext_mask, onlyspecs = False):
        with torch.no_grad():
            # Do prediction for one batch(each tile is devided in 6 batches) in the timestep
            input_batch, ref_map = reco_model.get_batch(test_idxs[0], test_mode=True, rl_depth=pred_depths)
            preds = self.unet(input_batch).detach().cpu()
            for batch_i in test_idxs[1:]:
                input_batch, _ = reco_model.get_batch(batch_i, test_mode=True, rl_depth=pred_depths)
                preds = torch.cat((preds, self.unet(input_batch).detach().cpu()), dim=0)
            del _
            torch.cuda.empty_cache()
            preds = preds.to(reco_model.device)   
    
            logger.info(f"Preds shape: {preds.shape}")
            logger.info(f"Ref map shape: {ref_map.shape}")
            depth_map = reco_model.reconstruct_full_map(preds)
            torch.cuda.empty_cache()
            ref_map[ref_map <0.05] = 0
            depth_map[depth_map <0.05] = 0
            return depth_map, ref_map
            
    def load_model_from_file(self, model_file):
        checkpoint  = torch.load(model_file)
        model_structue  = checkpoint['model_structure'] 
        model = UNet(encoder_channels= model_structue, decoder_channels=model_structue[:0:-1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()   
        return model

    def load_cnn_model_from_file(self, model_file):
        checkpoint  = torch.load(model_file)
        model_structue = checkpoint['model_strcture']
        if 'seq_h' not in checkpoint:
            seq_h = self.seq_h
        else:
            seq_h = checkpoint['seq_h']
        model = CNN1DSequential(model_structue, seq_h)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
        return model
    
    def load_cnn_models(self, cnn_model_files: dict):
        cnn_models = {}
        for rl_group, model_file in cnn_model_files.items():
            cnn_model  = self.load_cnn_model_from_file(model_file)
            cnn_models[rl_group] = cnn_model
        return cnn_models
    
def validate_reconstruction(sampling_distance, cluster_size, run_dir):
    unet_model_file  = find_model_file(UNET_MODEL_NAME)
    cnn_models = find_1dcnn_models(sampling_distance, cluster_size)

    if unet_model_file is None:
        raise ValueError(f"Model file not found for sampling distance {sampling_distance}")
    run_dir = os.path.join(run_dir)
    
    os.makedirs(run_dir, exist_ok=True)
    reco_model = ReconstructionModule(unet_model_file, cnn_models, sampling_distance, cluster_size, run_dir)
    return reco_model.reconstruct()

def find_1dcnn_models(sampling_distance, cluster_size):
    metadata_file = os.path.join(RUN_DIR, USRR_1DCNN_V1, "run_metadata.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} not found.")
    metadata_df = pd.read_csv(metadata_file)
    
    # Sampling_distance and cluster_size are in the metadata. Filter the rows based on these values
    filtered_df = metadata_df[(metadata_df['sampling_dist'] == sampling_distance) & 
                               (metadata_df['cluster_size'] == cluster_size)]
    
    # group by sampling_dist, cluster_size and rl_group and get the latest model file by run_id
    latest_model_files = {}
    for _, group in filtered_df.groupby(['rl_group']):
        group_id =  group['rl_group'].iloc[0]
        latest_run_id = group['run_id'].sort_values(ascending=False).iloc[0]
        latest_model_file = group[group['run_id'] == latest_run_id]['model_file'].values[0]
        latest_model_files[group_id] = latest_model_file
        
    return latest_model_files
