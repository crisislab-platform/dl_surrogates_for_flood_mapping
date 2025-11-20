from modules.lib.constants import OUTPUT_DIR, RUN_DIR, CARLISLE_DATA_DIR, SIMULATION_DATA_DIR
from modules.models.usrr_1dcnn.lib.gdal_lib  import gdal_asarray, read_shp_point, coords2rc, gdal_transform, gdal_writetiff
from modules.utils.run_util import check_device
from modules.models.usrr_1dcnn.unet import UNet
from modules.models.usrr_1dcnn.cnn1d import CNN1DSequential
from modules.datamanager.raster.raster_loader_usrr import ReconsturctionDataManager
from modules.datamanager.point.sequential_loader_1dcnn import CNNSequentialDataManager
from modules.models.usrr_1dcnn.unet import model_name as UNET_MODEL_NAME
from modules.lib.constants import USRR_1DCNN_V1, GRAPH_OUTPUT_DIR, SIMULATION_DATA_DIR, USRR_CNN1D_COMBINED
from modules.model_runner.model_utils import find_model_file
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map
from torch.utils.flop_counter import FlopCounterMode


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
import psutil


logger = logging.getLogger("ReconstructionModule")
logger.setLevel(logging.INFO)


class ReconstructionModule():
    
    def __init__(self, sampling_distance, cluster_size, run_dir, batch_size):
        
        # Initialize directories and files
        self.dem_asc_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
        self.simulation_dir  =  SIMULATION_DATA_DIR
        self.rep_loc_file_path = os.path.join(OUTPUT_DIR, "rls", f"rl_{sampling_distance}.asc")
        self.device =  check_device()
        self.run_dir = run_dir
        
        # Initialise parameters
        self.input_time_len_h = 9
        self.seq_h = self.input_time_len_h * 4
        self.sampling_distance = sampling_distance
        self.cluster_size = cluster_size
        
        #Load models
        self.cnn_model_files, self.cnn_train_history = find_1dcnn_models(sampling_distance, cluster_size)
        self.unet_model_file, self.unet_train_history  = find_model_file(UNET_MODEL_NAME)

        # Load models
        self.unet = self.load_unet_model_from_file(self.unet_model_file).to(self.device)
        self.cnn_models = self.load_cnn_models(self.cnn_model_files)

        
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
        logger.info("Predicting depths at representative locations using CNN models")
        pred_depths, reference_outputs, pred_time, cnn_profile = self.predict_rl_depth()
        logger.info(f"Time taken for RL depth prediction: {pred_time}")
        metrics = self.single_reconstruction(test_idxs, pred_depths, reference_outputs, cnn_profile)
        return metrics
      
    def multi_reconstruction(self, test_idxs, pred_depths, reference_outputs, cnn_profile):
        start_time = time.time()
        all_preds = []
        all_reference_maps = []
        all_tp = 0
        all_tn = 0
        all_fp = 0
        all_fn = 0
        
        # Process the first map to get unet profile
        pred_depths_map = pred_depths[0]
        depth_map, ref_map, prof_analysis_results = self.single_construct(0, pred_depths_map)
        unet_profile = prof_analysis_results
        overall_mse = 0
        overall_mrmse= 0
        overall_rmse = 0
        batches_of_test_indices = np.array_split(test_idxs, max(1, len(test_idxs)//20))
        
        for batch_indices in batches_of_test_indices:
            pred_maps, reference_maps = self.multiple_construct(batch_indices, pred_depths)

        
            logger.info(f"Depth map shape: {depth_map.shape}")
            logger.info(f"Reference map shape: {ref_map.shape}")       

            # overall_mse = self.loss_fn(pred_maps, reference_maps)
            # self.save_predictions(depth_map, map_i)
            # nse = self.nse_fn(ref, depth_map)
            overall_mse += self.loss_fn(pred_maps, reference_maps)
            overall_mrmse += self.mRMSE_fn(pred_maps, reference_maps)
            overall_rmse += np.sqrt(self.loss_fn(pred_maps, reference_maps).item())
               
            # create confusion matrix for flood/no flood
            threshold = 0.3
            pred_binary = (depth_map > threshold).float()
            ref_binary = (ref_map > threshold).float()
            
            all_preds.extend(pred_maps)
            all_reference_maps.extend(reference_maps)

            tp = ((pred_binary == 1) & (ref_binary == 1)).sum().item()
            tn = ((pred_binary == 0) & (ref_binary == 0)).sum().item()
            fp = ((pred_binary == 1) & (ref_binary == 0)).sum().item()
            fn = ((pred_binary == 0) & (ref_binary == 1)).sum().item()
            all_tp += tp
            all_tn += tn
            all_fp += fp
            all_fn += fn
            logger.info(f"Batch MSE {self.loss_fn(pred_maps, reference_maps).item()} mRMSE {self.mRMSE_fn(pred_maps, reference_maps)} all_tp {tp} all_tn {tn} all_fp {fp} all_fn {fn}")

        end_time = time.time()
        reconstruction_time = end_time - start_time
        
        mse = overall_mse / len(batches_of_test_indices)
        rmse = overall_rmse / len(batches_of_test_indices)
        mRMSE = overall_mrmse / len(batches_of_test_indices)
        
        # Calculate Hit Ratio and Critical Success Index (CSI)
        hit_ratio = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn) if (all_tp + all_tn + all_fp + all_fn) > 0 else 0
        csi = all_tp / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
        
        # F2 Score
        f2_score = (all_tp - all_fn) / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
                
        # F3 Score
        f3_score = (all_tp - all_fp) / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
        
        # Find max Memory Consumption out of cnn and unet models
        if cnn_profile is not None:
            max_cuda_memor_cnn = cnn_profile['max_cuda_memory']
            max_cpu_memory_cnn = cnn_profile['max_cpu_memory']
        if unet_profile is not None:
            max_cuda_memory_unet = unet_profile['max_cuda_memory']
            max_cpu_memory_unet = unet_profile['max_cpu_memory']
    
        pred_memory_usage = {
            "max_cuda_memory": max(max_cuda_memor_cnn, max_cuda_memory_unet) if cnn_profile and unet_profile else 0,
            "max_cpu_memory": max(max_cpu_memory_cnn, max_cpu_memory_unet) if cnn_profile and unet_profile else 0,
            "total_cpu_time": cnn_profile['total_cpu_time'] + unet_profile['total_cpu_time'] if cnn_profile and unet_profile else 0,
            "total_gpu_time": cnn_profile['total_gpu_time'] + unet_profile['total_gpu_time'] if cnn_profile and unet_profile else 0,
            "total_cpu_memory": 0,
            "total_gpu_memory": 0 
        }
        
        logger.info(f"Overall MSE: {mse}, RMSE: {rmse},  mRMSE: {mRMSE} Hit Ratio: {hit_ratio}, CSI: {csi}, F2 Score: {f2_score}, F3 Score: {f3_score}")
        flops = self.cnn_train_history['flops'] + unet_profile['reconstruct_flops']
        logger.info(f"Total FLOPs: {format_flops(flops)}")
        
        # Save the prediction maps
        # self.save_predictions_at_points(all_preds, all_reference_maps)
        metrics  = {
            'mse': mse,
            'rmse': rmse,
            'mRMSE': mRMSE,
            'nse': "",
            'pred_time': reconstruction_time,
            'flops': flops,
            "pred_memory_usage": pred_memory_usage , 
            'hit_rate': hit_ratio,
            'csi': csi,
            'f2_score': f2_score,
            'f3_score': f3_score
        }
        return metrics
    
    
    def single_reconstruction(self, test_idxs, pred_depths, reference_outputs, cnn_profile):
        start_time = time.time()
        all_preds = []
        all_reference_maps = []
        all_tp = 0
        all_tn = 0
        all_fp = 0
        all_fn = 0
        overall_mse = 0
        overall_nse = 0
        overall_mRMSE = 0  
        for map_i in range(len(test_idxs)):
            logger.info(f"Processing map {map_i}")           
            pred_depths_map = pred_depths[map_i]
            reference_outputs_map = reference_outputs[map_i]
            depth_map, ref_map, prof_analysis_results = self.single_construct(map_i, pred_depths_map)
            if map_i == 0 and prof_analysis_results is not None:
                unet_profile = prof_analysis_results
            # ref_map_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{(map_i + 76):04d}.wd")
            # ref_map = gdal_asarray(ref_map_file)
            # ref_map = torch.from_numpy(ref_map).float().to(self.device)
            # #set values below 0.3 to 0
            # ref_map[ref_map < 0.3] = 0
        
            logger.info(f"Depth map shape: {depth_map.shape}")
            logger.info(f"Reference map shape: {ref_map.shape}")       

            loss = self.loss_fn(depth_map, ref_map)
            
            # self.save_predictions(depth_map, map_i)
            nse = self.nse_fn(ref_map, depth_map)
            mRMSE = self.mRMSE_fn(depth_map, ref_map)
            overall_nse += nse
            overall_mse += loss.item()
            overall_mRMSE += mRMSE
            
            # create confusion matrix for flood/no flood
            threshold = 0.3
            pred_binary = (depth_map > threshold).float()
            ref_binary = (ref_map > threshold).float()
            
            tp = ((pred_binary == 1) & (ref_binary == 1)).sum().item()
            tn = ((pred_binary == 0) & (ref_binary == 0)).sum().item()
            fp = ((pred_binary == 1) & (ref_binary == 0)).sum().item()
            fn = ((pred_binary == 0) & (ref_binary == 1)).sum().item()
            
            all_tp += tp
            all_tn += tn
            all_fp += fp
            all_fn += fn
            
            logger.info(f"Map {map_i} MSE {loss.item()} NSE {nse} mRMSE {mRMSE} all_tp {tp} all_tn {tn} all_fp {fp} all_fn {fn}")
            all_preds.append(depth_map)
            all_reference_maps.append(ref_map)
        
        end_time = time.time()
        mse = overall_mse / test_idxs.shape[0]
        rmse = np.sqrt(mse)
        mRMSE = overall_mRMSE / test_idxs.shape[0]
        nse = overall_nse / test_idxs.shape[0]
        reconstruction_time = end_time - start_time
        
        # Calculate Hit Ratio and Critical Success Index (CSI)
        hit_ratio = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn) if (all_tp + all_tn + all_fp + all_fn) > 0 else 0
        csi = all_tp / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
        
        # F2 Score
        f2_score = (all_tp - all_fn) / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
                
        # F3 Score
        f3_score = (all_tp - all_fp) / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
        
        # Find max Memory Consumption out of cnn and unet models
        if cnn_profile is not None:
            max_cuda_memor_cnn = cnn_profile['max_cuda_memory']
            max_cpu_memory_cnn = cnn_profile['max_cpu_memory']
        if unet_profile is not None:
            max_cuda_memory_unet = unet_profile['max_cuda_memory']
            max_cpu_memory_unet = unet_profile['max_cpu_memory']
    
        pred_memory_usage = {
            "max_cuda_memory": max(max_cuda_memor_cnn, max_cuda_memory_unet) if cnn_profile and unet_profile else 0,
            "max_cpu_memory": max(max_cpu_memory_cnn, max_cpu_memory_unet) if cnn_profile and unet_profile else 0,
            "total_cpu_time": cnn_profile['total_cpu_time'] + unet_profile['total_cpu_time'] if cnn_profile and unet_profile else 0,
            "total_gpu_time": cnn_profile['total_gpu_time'] + unet_profile['total_gpu_time'] if cnn_profile and unet_profile else 0,
            "total_cpu_memory": 0,
            "total_gpu_memory": 0 
        }
        
        logger.info(f"Overall MSE: {mse}, RMSE: {rmse}, NSE: {nse} mRMSE: {mRMSE} Hit Ratio: {hit_ratio}, CSI: {csi}, F2 Score: {f2_score}, F3 Score: {f3_score}")
        flops = max(self.cnn_train_history['flops'], self.unet_train_history['flops'])
        logger.info(f"Total FLOPs: {format_flops(flops)}")
        
        # Save the prediction maps
        # self.save_predictions_at_points(all_preds, all_reference_maps)
        metrics  = {
            'mse': mse,
            'rmse': rmse,
            'mRMSE': mRMSE,
            'nse': nse,
            'pred_time': reconstruction_time,
            'flops': flops,
            "pred_memory_usage": pred_memory_usage , 
            'hit_rate': hit_ratio,
            'csi': csi,
            'f2_score': f2_score,
            'f3_score': f3_score
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
        
        #Run one of the CNN models to get memory profiling
                # Add profiler for the parallel execution part
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
                    record_shapes=True, with_stack=True) as prof_rl:
            rl_group = list(self.cnn_models.keys())[0]
            
            model = self.cnn_models[rl_group]
            _, _  = self.get_rl_group_predictions(rl_group, model)
            # Log profiler results for RL depth prediction
            logger.info("RL Depth Prediction Profiler Results:")
            logger.info(prof_rl.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            prof_analsys_results = profiler_analysis(prof_rl.key_averages())
            
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
        
        # Add thread lock for concurrent access to all_preds
        import threading
        preds_lock = threading.Lock()
        
        #Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
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
                    # Process each timestep with thread synchronization
                    with preds_lock:  # Acquire lock before modifying shared data
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
        return all_preds, reference_output, pred_time, prof_analsys_results
        
    def get_rl_group_predictions(self, rl_group, model):
        try: 
            total_prediction_time = 0  # Initialize variable used in the method
            cnn_data_manager = CNNSequentialDataManager(16, self.input_time_len_h,
                                                rl_group, self.sampling_distance, 
                                                self.cluster_size, tuning_mode=False,
                                                reconstruction_mode=True, 
                                                reco_data_manager=self.data_manager)
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
        except Exception as exc:
                logger.error(f"RL group {rl_group} generated an exception: {exc}")
        
        
                    
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
            prof_analysis_results = None
            # Only profile the first map/timestep
            if map_i == 0:
                input_batch, output_batch, reference_map = self.data_manager.get_batch(map_i, pred_depths)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
                         with_flops=True) as prof_construct:
                    preds = self.unet(input_batch)
                
                preds = preds.squeeze(1)
                loss = self.loss_fn(preds, output_batch)
                logger.info(f"Map {map_i} Loss: {loss.item()}")
                
                process = psutil.Process()
                start_cpu_time = process.cpu_times().user + process.cpu_times().system
                start_time = time.time()
                mem_before = process.memory_info().rss / (1024 * 1024)
                
                # Need to measure flops for this one
                predicted_map = self.data_manager.reconstruct_full_map(preds)
                
                mem_after = process.memory_info().rss / (1024 * 1024)
                memory_used = mem_after - mem_before
                logger.info(f"Memory used during UNet reconstruction: {memory_used:.2f} MB")
                
                end_time = time.time()
                end_cpu_time = process.cpu_times().user + process.cpu_times().system
                
                # Log profiler results for single construct
                logger.info("Single Construct (First Timestep) Profiler Results:")
                logger.info(prof_construct.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
                prof_analysis_results = profiler_analysis(prof_construct.key_averages())
                logger.info(f"Profiler Analysis Results for single_construct: {prof_analysis_results}")
                logger.info(f"Memory usage for single_construct: {prof_analysis_results['max_cpu_memory']} MB")
                logger.info(f"CUDA Memory usage for single_construct: {prof_analysis_results['max_cuda_memory']} MB")
                logger.info(f"FLOPs for single_construct: {format_flops(prof_analysis_results['flops'])}")   
                     
            else:
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
            return predicted_map, reference_map, prof_analysis_results
        
    def multiple_construct(self, map_indices, pred_depths):
         with torch.no_grad():
            self.unet.eval()
            input_batch, output_batch, reference_maps = self.data_manager.get_batch_multiple(map_indices, pred_depths)
            preds_multiple= self.unet(input_batch)
            preds_multiple = preds_multiple.squeeze(1)
            loss = self.loss_fn(preds_multiple, output_batch)
            logger.info(f"Map {map_indices} MSE: {loss.item()}")
            
            predicted_maps = []
            index = 0
            for preds in preds_multiple:
                logger.info(f"Reconstructing map {index}")
                predicted_map = self.data_manager.reconstruct_full_map(preds)
                torch.cuda.empty_cache()
                reference_maps[reference_maps < 0.3] = 0
                predicted_map[predicted_map < 0.3] = 0
                predicted_maps.append(predicted_map)
                index += 1
            
            # predicted_map[reference_map < 0.3] = 0
            return predicted_maps, reference_maps
        
        
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
        self.input_time_len_h = checkpoint.get('input_time_len_h', 9)
        self.seq_h = self.input_time_len_h * 4
        convo_kernel = checkpoint.get('convo_kernel', 4)
        pool_kernel = checkpoint.get('pool_kernel', 3)
        model = CNN1DSequential(model_structue, self.seq_h, convo_kernel, pool_kernel).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def load_cnn_models(self, cnn_model_files: dict):
        cnn_models = {}
        for rl_group, model_file in cnn_model_files.items():
            cnn_model  = self.load_cnn_model_from_file(model_file)
            cnn_models[rl_group] = cnn_model
        return cnn_models
    
    def save_predictions(self, pred, idx):
        output_dir = os.path.join(RUN_DIR, "output_maps", USRR_CNN1D_COMBINED)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving prediction map to {output_dir}")
        save_prediction_map(pred, output_dir, idx, USRR_CNN1D_COMBINED)
        
    def save_predictions_at_points(self, predictions, ground_truth):
        """
        Save model predictions at specific points of interest to a CSV file
        
        Args:
            predictions: Model predictions tensor
            ground_truth: Ground truth tensor
            points_csv: Path to the CSV file containing points of interest
        """
        try:
            points_csv = os.path.join(OUTPUT_DIR, "points_of_interest.csv")
            logger.info(f"Saving predictions at points of interest from {points_csv}")
            
            # Read points of interest
            poi_df = pd.read_csv(points_csv)
            
            # Convert tensors to numpy arrays for easier handling
            pred_np = [pred.cpu().numpy() for pred in predictions]
            truth_np = [truth.cpu().numpy() for truth in ground_truth]
            
            #convert to numpy arrays
            pred_np = np.array(pred_np)
            truth_np = np.array(truth_np)
            
            ref_file = os.path.join(SIMULATION_DATA_DIR, "Run1-0000.wd")
            with rasterio.open(ref_file) as src:
                height = src.height
                width = src.width
                profile = src.profile
                transform = src.transform
                crs = src.crs
                
            pred_np = pred_np.reshape(pred_np.shape[0],height, width)
            truth_np = truth_np.reshape(truth_np.shape[0], height, width)
           
            # Extract predictions and ground truth at each point
            results = []
            timesteps = min(pred_np.shape[0], truth_np.shape[0])
            
            for _, point in poi_df.iterrows():
                point_id = point['Point_ID']
                row = int(point['Row'])
                col = int(point['Column'])
                elev = point['Elevation_m']
                label = point['Label']
                percentile = point['Percentile']
                
                # For each timestep, get the prediction and ground truth at this point
                for t in range(timesteps):
                    # Check if indices are in bounds
                    if (t < pred_np.shape[0] and 
                        row < pred_np.shape[1] and 
                        col < pred_np.shape[2]):
                        
                        pred_depth = pred_np[t, row, col]
                        true_depth = truth_np[t, row, col]
                        
                        results.append({
                            'Point_ID': point_id,
                            'Label': label,
                            'Percentile': percentile,
                            'Row': row,
                            'Column': col,
                            'Elevation_m': elev,
                            'Timestep': t,
                            'Predicted_Depth_m': pred_depth,
                            'True_Depth_m': true_depth,
                            'Error_m': pred_depth - true_depth,
                            'Model_Name': USRR_CNN1D_COMBINED,
                            'Run_ID': USRR_CNN1D_COMBINED
                        })
            
            # Create DataFrame and save to CSV
            results_df = pd.DataFrame(results)
            output_path = os.path.join(RUN_DIR, USRR_CNN1D_COMBINED,  f"predictions_at_points_final.csv")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved point predictions to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving predictions at points: {e}")
    
    
def find_1dcnn_models(sampling_distance, cluster_size):
    metrics_file = os.path.join(RUN_DIR, USRR_1DCNN_V1, "final_training_metrics.csv")
    performance_file = os.path.join(RUN_DIR, USRR_1DCNN_V1, "final_performance_metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metadata file {metrics_file} not found.")
    metrics_df = pd.read_csv(metrics_file)
    perf_df = pd.read_csv(performance_file)
    
    
    filtered_df = metrics_df[metrics_df['hyperparameters'].apply(
        lambda x: eval(x).get('sampling_dist') == sampling_distance and 
                  eval(x).get('n_clusters') == cluster_size
    )]
    
    # Extract rl_group from hyperparameters column
    filtered_df['rl_group'] = filtered_df['hyperparameters'].apply(lambda x: eval(x).get('rl_group', 'unknown'))
    
    # group by sampling_dist, cluster_size and rl_group and get the latest model file by run_id
    latest_model_files = {}
    total_flops = 0
    total_neurons = 0
    total_training_time = 0
    max_cuda_memory = 0
    max_cpu_memory = 0
    total_cpu_time = 0
    total_gpu_time = 0
    total_params = 0
    for rl_group in filtered_df['rl_group'].unique():
        group = filtered_df[filtered_df['rl_group'] == rl_group]
        latest_run_id = group['run_id'].sort_values(ascending=False).iloc[0]
        latest_model_file = group[group['run_id'] == latest_run_id]['model_file'].values[0]
        perf_row = perf_df[perf_df['run_id'] == latest_run_id]
        traing_row = metrics_df[metrics_df['run_id'] == latest_run_id]
        if not perf_row.empty:
            total_flops += perf_row['flops'].values[0]
            total_neurons += traing_row['total_neurons'].values[0]
            total_training_time = traing_row['train_time'].values[0]
            total_params = traing_row['trainable_params'].values[0]
            memory_usage = traing_row['training_memory_usage'].values[0]
            
            memory_usage = eval(memory_usage)
            max_cuda_memory = max(max_cuda_memory, memory_usage['max_cuda_memory'])
            max_cpu_memory = max(max_cpu_memory, memory_usage['max_cpu_memory'])
            total_cpu_time += memory_usage['total_cpu_time']
            total_gpu_time += memory_usage['total_gpu_time']
            
        latest_model_files[rl_group] = latest_model_file
        
    cnn_history = {
        'flops': total_flops,
        'num_rls': len(latest_model_files),
        'total_neurons':total_neurons,
        'train_time': total_training_time,
        'memory': {
            'max_cuda_memory': max_cuda_memory,
            'max_cpu_memory': max_cpu_memory, 
            'total_cpu_time': total_cpu_time, 
            'total_gpu_time': total_gpu_time,
            'total_cuda_memory': 0,
            'total_cpu_memory': 0,
        },
        'trainable_params': total_params,
    }
        
    return latest_model_files, cnn_history
    

