from modules.lib.constants import OUTPUT_DIR, RUN_DIR, CARLISLE_DATA_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib  import gdal_asarray, read_shp_point, coords2rc, gdal_transform, gdal_writetiff
from modules.utils.run_util import check_device
from modules.models.usrr_1dcnn.unet import UNet
from modules.models.usrr_1dcnn.cnn1d import CNN1DSequential
from modules.datamanager.raster.raster_loader_usrr import ReconsturctionDataManager
from modules.datamanager.point.sequential_loader_1dcnn import CNNSequentialDataManager
from modules.models.usrr_1dcnn.unet import model_name as UNET_MODEL_NAME
from modules.lib.constants import USRR_1DCNN_V1, GRAPH_OUTPUT_DIR, SIMULATION_DATA_DIR
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


logger = logging.getLogger("ReconstructionModule")
logger.setLevel(logging.INFO)

def reconstruct_and_test(sampling_distance, cluster_size, run_dir, batch_size):
    cnn_model_files = find_1dcnn_models(sampling_distance, cluster_size)
    unet_model_file  = find_model_file(UNET_MODEL_NAME)

    if unet_model_file is None:
        raise ValueError(f"Model file not found for sampling distance {sampling_distance}")
    
    run_dir = os.path.join(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    reco_module = ReconstructionModule(unet_model_file, cnn_model_files, sampling_distance, cluster_size, run_dir, batch_size)
    return reco_module.reconstruct()

class ReconstructionModule():
    
    def __init__(self, uent_model_file, cnn_model_files, sampling_distance, cluster_size, run_dir, batch_size):
        
        # Initialize directories and files
        self.dem_asc_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
        self.simulation_dir  =  SIMULATION_DATA_DIR
        self.rep_loc_file_path = os.path.join(OUTPUT_DIR, "rls", f"rl_{sampling_distance}.asc")
        self.device =  check_device()
        self.run_dir = run_dir
        
        # Initialise parameters
        self.input_time_len_h = 12
        self.seq_h = self.input_time_len_h * 4
        self.sampling_distance = sampling_distance
        self.cluster_size = cluster_size

        # Load models
        self.unet = self.load_unet_model_from_file(uent_model_file).to(self.device)
        self.cnn_models = self.load_cnn_models(cnn_model_files)
        
        # Test variables
        self.test_event_ids = [1]
        self.loss_fn = nn.MSELoss()
        self.mrmse_fn = nn.MSELoss(reduction='mean')
        self.batch_size = batch_size
        
        #Load data manager
        self.data_manager = ReconsturctionDataManager(sampling_dist=self.sampling_distance, run_dir=self.run_dir, batches_per_map=self.batch_size)
    
    def reconstruct(self):
        logger.info("Running reconstruction")
        test_idxs = self.data_manager.reconstruction_test_idxs
        logger.info(f"Testing the recosntruction for : {test_idxs.shape} maps")
    
        overall_mse = 0
        overall_nse = 0
        overall_mRMSE = 0
        start_time = time.time()
        
        logger.info("Predicting depths at representative locations using CNN models")
        pred_depths, reference_outputs, pred_time = self.predict_rl_depth()
        logger.info(f"Time taken for RL depth prediction: {pred_time}")
        
        for map_i in range(len(test_idxs)):
            logger.info(f"Processing map {map_i}")           
            pred_depths_map = pred_depths[map_i]
            depth_map, ref_map = self.single_construct(map_i, pred_depths_map)
            
            # ref_map_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{(map_i + 76):04d}.wd")
            # ref_map = gdal_asarray(ref_map_file)
            # ref_map = torch.from_numpy(ref_map).float().to(self.device)
            # #set values below 0.3 to 0
            # ref_map[ref_map < 0.3] = 0
        
            logger.info(f"Depth map shape: {depth_map.shape}")
            logger.info(f"Reference map shape: {ref_map.shape}")       

            loss = self.loss_fn(depth_map, ref_map)
            if map_i == 150:
                self.visualise_error_distribution(depth_map, ref_map)
            nse = self.nse_fn(ref_map, depth_map)
            mRMSE = self.mRMSE_fn(depth_map, ref_map)
            overall_nse += nse
            overall_mse += loss.item()
            overall_mRMSE += mRMSE
            logger.info(f"Map {map_i} MSE {loss.item()} NSE {nse} mRMSE {mRMSE} ")
            # if map_i == 145:
            #     self.save_depth_map(depth_map, map_i)
                
        end_time = time.time()
        mse = overall_mse / test_idxs.shape[0]
        rmse = np.sqrt(mse)
        mRMSE = overall_mRMSE / test_idxs.shape[0]
        nse = overall_nse / test_idxs.shape[0]
        reconstruction_time = end_time - start_time
        
        logger.info(f"Overall MSE: {mse}, RMSE: {rmse}, NSE: {nse} mRMSE: {mRMSE}")
        metrics  = {
            'mse': mse,
            'rmse': rmse,
            'mRMSE': mRMSE,
            'nse': nse,
            'pred_time': reconstruction_time,
            'wet_rmse': None,
            'wet_acc': None,
            'flops': None,
        }
        return metrics
    
    def visualise_error_distribution(self, depth_map, ref_map):
        """
        Visualize the error distribution between predicted depth map and reference map.
        Creates and saves visualizations showing error patterns.
        
        Args:
            depth_map (torch.Tensor): Predicted depth map
            ref_map (torch.Tensor): Reference/ground truth depth map
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        import os
        
        # Move tensors to CPU and convert to numpy arrays
        depth_map_np = depth_map.detach().cpu().numpy().flatten()
        
        # Load reference map from files
        ref_map_file = os.path.join(SIMULATION_DATA_DIR, "Run1-0226.wd")
        ref_map_np = gdal_asarray(ref_map_file).flatten()
        
        
        # Reshape to 2D for visualization
        depth_map_2d = depth_map.detach().cpu().numpy().reshape(self.dem_map.shape)
        ref_map_2d = ref_map_np.reshape(self.dem_map.shape)
        ref_map_2d[ref_map_2d < 0.3] = 0  # Set dry cells to 0 for visualization
        
        ref_map_other =  ref_map.detach().cpu().numpy().reshape(self.dem_map.shape)
        
        # Create error map
        error = (depth_map_np - ref_map_np)
        
        # Only consider points where reference depth is significant (wet cells)
        wet_mask = ref_map_np >= 0.3
        dry_mask = ref_map_np < 0.3
        
        error_wet = error[wet_mask]
        error_dry = error[dry_mask]
        
        # Create output directory
        vis_dir = os.path.join(GRAPH_OUTPUT_DIR)
        os.makedirs(vis_dir, exist_ok=True)

        # Reshape the error map to match the DEM dimensions
        full_error_map_wet = np.zeros_like(self.dem_map).flatten()
        full_error_map_dry = np.zeros_like(self.dem_map).flatten()  
        full_error_map_wet[wet_mask] = error_wet
        full_error_map_dry[dry_mask] = error_dry
        error_map_wet = full_error_map_wet.reshape(self.dem_map.shape)
        error_map_dry = full_error_map_dry.reshape(self.dem_map.shape)
        
        # Visualize predicted depth map
        plt.figure(figsize=(12, 10))
        im = plt.imshow(depth_map_2d, cmap='Blues', vmin=0, vmax=np.max(depth_map_2d) + 0.5)
        plt.colorbar(im, label='Depth [m]')
        plt.title('Predicted Flood Depth')
        plt.savefig(os.path.join(vis_dir, "predicted_depth_map.png"), dpi=300)
        logger.info(f"Saved predicted depth map to {vis_dir}/predicted_depth_map.png")
        plt.close()
        
        # Visualize reference depth map
        plt.figure(figsize=(12, 10))
        im = plt.imshow(ref_map_2d, cmap='Blues', vmin=0, vmax=np.max(ref_map_2d) + 0.5)
        plt.colorbar(im, label='Depth [m]')
        plt.title('Reference Flood Depth')
        plt.savefig(os.path.join(vis_dir, "reference_depth_map.png"), dpi=300)
        logger.info(f"Saved reference depth map to {vis_dir}/reference_depth_map.png")
        plt.close()
        
        plt.figure(figsize=(12, 10))
        im = plt.imshow(ref_map_other, cmap='Blues', vmin=0, vmax=np.max(ref_map_other) + 0.5)
        plt.colorbar(im, label='Depth [m]')
        plt.title('Reference Flood Depth')
        plt.savefig(os.path.join(vis_dir, "reference_depth_map_other.png"), dpi=300)
        logger.info(f"Saved reference depth map to {vis_dir}/reference_depth_map_other.png")
        plt.close()
        
        # Visualize error maps (as already implemented)
        plt.figure(figsize=(12, 10))
        plt.imshow(error_map_wet, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Error [m]')
        plt.title('Spatial Error Distribution')
        plt.savefig(os.path.join(vis_dir, "spatial_error_map_wet.png"), dpi=300)
        logger.info(f"Saved spatial error map to {vis_dir}/spatial_error_map.png")
        plt.close()
        
        plt.figure(figsize=(12, 10))
        plt.imshow(error_map_dry, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Error [m]')
        plt.title('Spatial Error Distribution (Dry Cells)')
        plt.savefig(os.path.join(vis_dir, "spatial_error_map_dry.png"), dpi=300)
        logger.info(f"Saved spatial error map to {vis_dir}/spatial_error_map_dry.png")
        plt.close()
    
    def find_representative_locations(self):
        #load demp map
        self.dem_map = gdal_asarray(self.dem_asc_file)
        self.rl_map = gdal_asarray(self.rep_loc_file_path)
        self.filter_mask  = self.rl_map == 1
        self.rl_filter = lambda x: x[self.filter_mask]
        
    def predict_rl_depth(self):
        self.find_representative_locations()
        
        # Create copies of each tensor
        reference_output = [tensor.clone() for tensor in self.data_manager.preloaded_maps]
        
        # For each reference map. Make non-zero values only at representative locations
        for i in range(len(reference_output)):
            ref_map = reference_output[i]
            ref_map[~self.filter_mask] = 0
            reference_output[i] = ref_map
            
        #Initialize empty list to store GPU tensors
        all_preds = [] 
        pred_start = time.time()
        
        #Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
                    # rl_locations = data_manager.coords_to_cluster_rls # [(1,2), (3,4)),..]
                    # # Get the first element of the tuple for each rl location in rl_locations
                    # row_indices = [rl[0] for rl in rl_locations]
                    # col_indices = [rl[1] for rl in rl_locations]
                    # Process each timestep
                    for i in range(y_hat_np.shape[0]):
                        # Initialize map for this timestep if needed
                        if len(all_preds) <= i:
                            all_preds.append(torch.zeros(self.dem_map.shape, device=self.device))
                        
                        # Add predictions to the map at the proper locations
                        all_preds[i][data_manager.filter_mask] = y_hat_np[i]
                    
                    logger.info(f"Completed predictions for RL group {rl_group}")
                except Exception as exc:
                    logger.error(f"RL group {rl_group} generated an exception: {exc}")
        pred_end = time.time()
        pred_time = pred_end - pred_start
        logger.info(f"Total prediction time: {pred_time:.4f} seconds")
        logger.info(f"Completed predictions for {len(all_preds)} RL groups in parallel")
        return all_preds, reference_output, pred_time
        
    def get_rl_group_predictions(self, rl_group, model):
        total_prediction_time = 0  # Initialize variable used in the method
        cnn_data_manager = CNNSequentialDataManager(32, self.input_time_len_h,
                                             rl_group, self.sampling_distance, 
                                             self.cluster_size, tuning_mode=False, reconstruction_mode=True, reco_data_manager=self.data_manager)
        with torch.no_grad():
            model.eval()
            test_inputs = cnn_data_manager.test_input
            start_time = time.time()    
            y_hat = model(test_inputs.float())
            end_time = time.time()
            prediction_time = end_time - start_time
            total_prediction_time += prediction_time
            logger.info(f"Prediction time for {rl_group}: {prediction_time:.4f} seconds")
            return y_hat, cnn_data_manager
                    
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
    
    def mRMSE_fn(self, pred, ref_out):
        # Define threshold for wet cells
        threshold = 0.3
        
        # Create binary mask
        ref_wet = (ref_out > threshold).float()
        
        # Calculate RMSE for wet cells only
        pred_wet_values = pred * ref_wet
        ref_wet_values = ref_out * ref_wet
        wet_loss = self.loss_fn(pred_wet_values, ref_wet_values)
        rmse_wet = np.sqrt(wet_loss.item())
        logger.info(f"Wet cells RMSE: {rmse_wet}")
        return rmse_wet
                       
    def single_construct(self, map_i, pred_depths):
        with torch.no_grad():
            self.unet.eval()
            input_batch, output_batch, reference_map = self.data_manager.get_batch(map_i, pred_depths)
            preds = self.unet(input_batch)
            preds = preds.squeeze(1)
            loss = self.loss_fn(preds, output_batch)
            logger.info(f"Map {map_i} Loss: {loss.item()}")
            predicted_map = self.data_manager.reconstruct_full_map(preds)
            torch.cuda.empty_cache()
            reference_map[reference_map < 0.3] = 0
            predicted_map[predicted_map < 0.3] = 0
            
            # predicted_map[reference_map < 0.3] = 0
            
            return predicted_map, reference_map
        
        
    def load_unet_model_from_file(self, model_file):
        checkpoint  = torch.load(model_file)
        model_structue  = [2, 32, 64, 128, 256] 
        model = UNet(encoder_channels= model_structue, decoder_channels=model_structue[:0:-1]).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() 
        model = model.to(self.device)
        return model

    def load_cnn_model_from_file(self, model_file):
        checkpoint  = torch.load(model_file)
        model_structue = checkpoint['model_structure']
        if 'seq_h' not in checkpoint:
            seq_h = self.seq_h
        else:
            seq_h = checkpoint['seq_h']
        convo_kernel = checkpoint.get('convo_kernel', 4)
        pool_kernel = checkpoint.get('pool_kernel', 3)
        model = CNN1DSequential(model_structue, seq_h, convo_kernel, pool_kernel).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def load_cnn_models(self, cnn_model_files: dict):
        cnn_models = {}
        for rl_group, model_file in cnn_model_files.items():
            cnn_model  = self.load_cnn_model_from_file(model_file)
            cnn_models[rl_group] = cnn_model
        return cnn_models
    
def find_1dcnn_models(sampling_distance, cluster_size):
    metrics_file = os.path.join(RUN_DIR, USRR_1DCNN_V1, "final_training_metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metadata file {metrics_file} not found.")
    metrics_df = pd.read_csv(metrics_file)
    
    filtered_df = metrics_df[metrics_df['hyperparameters'].apply(
        lambda x: eval(x).get('sampling_dist') == sampling_distance and 
                  eval(x).get('n_clusters') == cluster_size
    )]
    
    # Extract rl_group from hyperparameters column
    filtered_df['rl_group'] = filtered_df['hyperparameters'].apply(lambda x: eval(x).get('rl_group', 'unknown'))
    
    # group by sampling_dist, cluster_size and rl_group and get the latest model file by run_id
    latest_model_files = {}
    for rl_group in filtered_df['rl_group'].unique():
        group = filtered_df[filtered_df['rl_group'] == rl_group]
        latest_run_id = group['run_id'].sort_values(ascending=False).iloc[0]
        latest_model_file = group[group['run_id'] == latest_run_id]['model_file'].values[0]
        latest_model_files[rl_group] = latest_model_file
        
    return latest_model_files


