from modules.lib.constants import SIMULATION_DATA_DIR, OUTPUT_DIR, RUN_DIR
from modules.models.srr_lstm.srr.sdr_algorithm_reco import reconstruct_flood_inundation_map
from modules.datamanager.raster.raster_loader_lstmsrr import ReconsturctionDataManager
from modules.datamanager.point.sequential_loader_lstm import LSTMSequentialDataManager
from modules.lib.constants import LSTM_SRR_V1
from modules.models.srr_lstm.srr_lstm import LSTMModel
from modules.models.srr_lstm.srr.gdal_func import gdal_asarray, rc2coords, gdal_transform
from modules.utils.run_util import check_device
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map

import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import concurrent.futures

import time
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SDRReconstructor")

class SDRReconstructor():
    def __init__(self):
        
        self.dem_asc_file = f"{SIMULATION_DATA_DIR}/Carlisle_5m.asc"

        self.max_inundation_timestep = {
            'event_id': 3,
            'timestep': 94
        }
        
        #configs
        self.input_time_len_h = 12
        self.device = check_device()
        
        # Files and directories
        self.max_inundation_file = f"{SIMULATION_DATA_DIR}/Run{self.max_inundation_timestep['event_id']}-{(self.max_inundation_timestep['timestep']):04d}.wd"
        self.stopping_values_dem = [-555]
        self.reduction_result_dir = os.path.join(OUTPUT_DIR, "sdr_reduction_results")
        self.reco_result_dir = os.path.join(OUTPUT_DIR, "sdr_reconstruction_results")
        self.rls_shp_file = os.path.join(self.reduction_result_dir, "representative_locations.shp")
        self.sdr_thalwegs_file = os.path.join(self.reduction_result_dir, "sdr_results_thalwegs.shp")
        self.demfile = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
        
        self.data_manager = ReconsturctionDataManager()
        self.loss_fn = nn.MSELoss()
        self.mRMSE_fn = self.mRMSE_fn
        self.nse_fn = self.nse_fn

        # Predict water depth using LSTM models
        self.lstm_models, self.lstm_metrics = self.find_lstm_models()
        self.num_rls = len(self.lstm_models)
        self.find_coords_of_modelling_domain(self.demfile)
        # self.x_coords = np.arange(0,951)
        # self.y_coords = np.arange(0,611)
        
    def find_coords_of_modelling_domain(self, demfile):
        """
        Find the coordinates of the modelling domain from the DEM file.
        Returns arrays of x and y coordinates in the spatial reference system.
        """
        # Get DEM data and transform
        dem_array = gdal_asarray(demfile)
        dem_transform = gdal_transform(demfile)
        
        # Get dimensions
        rows, cols = dem_array.shape
        target_xy_grid = []
        
        # Fill x coordinates (only need one row)
        for i in range(rows):
            for j in range(cols):
                x, y = rc2coords(dem_transform, (i, j))
                target_xy_grid.append((x, y))
    
        self.target_xy_grid = target_xy_grid
    
    def find_lstm_models(self):
        metrics_file = os.path.join(RUN_DIR, LSTM_SRR_V1, "final_training_metrics.csv")
        performance_file = os.path.join(RUN_DIR, LSTM_SRR_V1, "final_performance_metrics.csv")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metadata file {metrics_file} not found.")
        
        metrics_df = pd.read_csv(metrics_file)
        performance_df = pd.read_csv(performance_file)
        metrics_df['rl_group'] = metrics_df['hyperparameters'].apply(lambda x: eval(x).get('rl_group', 'unknown'))
        
        # Group by rl_group and get the latest model file by run_id
        latest_model_files = {}
        total_flops = 0
        total_nurons = 0
        total_training_time = 0
        total_params = 0
        max_cuda_memory = 0
        max_cpu_memory = 0
        total_cpu_time = 0
        total_gpu_time = 0
        
        for rl_id in metrics_df['rl_group'].unique():
            group = metrics_df[metrics_df['rl_group'] == rl_id]
            latest_run_id = group['run_id'].sort_values(ascending=False).iloc[0]
            latest_model_file = group[group['run_id'] == latest_run_id]['model_file'].values[0]
            latest_model_files[rl_id] = latest_model_file 
            
            # Get FLOPs for the latest run  
            perf_row = performance_df[performance_df['run_id'] == latest_run_id]
            training_row = metrics_df[metrics_df['run_id'] == latest_run_id]
            flops = perf_row['flops'].values[0] 
            total_training_time += training_row['train_time'].values[0]
            total_nurons += training_row['total_neurons'].values[0]
            
            memory_usage = training_row['training_memory_usage'].values[0]
            #convert memory usage from string to dict
            memory_usage = eval(memory_usage) if isinstance(memory_usage, str) else memory_usage
            
            total_flops += flops
            cpu_memory = memory_usage['max_cpu_memory']
            cuda_memory = memory_usage['max_cuda_memory']
            total_cpu_time += memory_usage['total_cpu_time']
            total_gpu_time += memory_usage['total_gpu_time']
            if cpu_memory > max_cpu_memory:
                max_cpu_memory = cpu_memory
            if cuda_memory > max_cuda_memory:
                max_cuda_memory = cuda_memory
            total_flops += flops
            total_params += training_row['trainable_params'].values[0]
            
        lstm_metrics = {
            'total_flops': total_flops,
            'num_rls': len(latest_model_files),
            'total_neurons':total_nurons,
            'train_time': total_training_time,
            'training_memory_usage': {
                'max_cuda_memory': max_cuda_memory,
                'max_cpu_memory': max_cpu_memory, 
                'total_cpu_time': total_cpu_time, 
                'total_gpu_time': total_gpu_time,
                'total_cuda_memory': 0,
                'total_cpu_memory': 0,
            },
            'trainable_params': total_params,
        }
        return latest_model_files, lstm_metrics

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
    
    def load_lstm_models(self):
        pass
        # logger.info("Loading LSTM models for SDR reconstruction...")

        # lstm_models = {}
        # for rl_id, model_file in model_files.items():
        #     if not os.path.exists(model_file):
        #         logger.warning(f"Model file {model_file} for RL group {rl_id} does not exist.")
        #         continue
        #     lstm_models[rl_id] = self.load_lstm_model_from_file(model_file)
        # self.num_rls = len(lstm_models)
        # return lstm_models
    
    def load_lstm_model_from_file(self, model_file):
        checkpoint  = torch.load(model_file)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        lstm_dim = checkpoint['lstm_dim']
        output_dim = checkpoint['output_dim']
        model = LSTMModel(input_dim, hidden_dim, lstm_dim, output_dim).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def predict_water_depths(self):
        
        #profiling memory usage for the first model
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
                           record_shapes=True, with_stack=True) as prof:
            rl_id = list(self.lstm_models.keys())[0]
            model_file = self.lstm_models[rl_id]
            logger.info(f"Loading model for RL group {rl_id} from {model_file}")
            _, _ = self.get_lstm_prediction(rl_id, model_file)
                
        logger.info(f"Memory profiling completed. Results saved to lstm_model_memory_usage.txt")
        logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        prof_analysis_results = profiler_analysis(prof.key_averages())
        logger.info(f"Memory usage for lstm: {prof_analysis_results['max_cpu_memory']} MB")
        logger.info(f"CUDA Memory usage for lstm: {prof_analysis_results['max_cuda_memory']} MB")
        
        #Initialize empty list to store GPU tensors
        number_of_maps = len(self.data_manager.reconstruction_test_idxs)
        all_preds = np.empty((number_of_maps, self.num_rls, 1), dtype=object)
        pred_start = time.time()
         
        #Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all prediction tasks
            future_to_group = {
                executor.submit(self.get_lstm_prediction, rl_id, model_file): rl_id
                for rl_id, model_file in self.lstm_models.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_group):
                rl_id = future_to_group[future]
                try:
                    y_hat_np, ref = future.result()
                    for i in range(ref.shape[0]):
                        depth = ref[i][0].item()
                        rl_idx = int(rl_id) - 1
                        all_preds[i][rl_idx][0] = depth
                    logger.info(f"Completed predictions for RL group {rl_id}")
                except Exception as exc:
                    logger.error(f"RL group {rl_id} generated an exception: {exc}")
                    
        pred_end = time.time()
        pred_time = pred_end - pred_start
        logger.info(f"Total prediction time: {pred_time:.4f} seconds")
        logger.info(f"Completed predictions for {len(all_preds)} RL groups in parallel")
        return all_preds, pred_time, prof_analysis_results
    
    def get_lstm_prediction(self, rl_id, model_file):
        model = self.load_lstm_model_from_file(model_file)
        lstm_data_manager = LSTMSequentialDataManager(32, self.input_time_len_h,
                                             rl_id, tuning_mode=False, reconstruction_mode=True, reco_data_manager=self.data_manager)
        with torch.no_grad():
            model.eval()
            test_inputs = lstm_data_manager.test_input
            test_outputs = lstm_data_manager.test_output
            start_time = time.time()    
            y_hat = model(test_inputs.float())
            end_time = time.time()
            prediction_time = end_time - start_time
            logger.info(f"Prediction time for {rl_id}: {prediction_time:.4f} seconds")
            return y_hat, test_outputs
    
    def process_single_map(self, map_i, test_idx, pred_depths_map, profile_memory=False):
        """Process a single map reconstruction"""
        logger.info(f"Processing map {map_i}")           
        pred_depths_list = pred_depths_map
        pred_depths_list = [depth[0] for depth in pred_depths_list]
        cpu_time = 0
        srr_profile = None
        
        if profile_memory:
            #cpu time
            process = psutil.Process(os.getpid())
            start = time.process_time()
            mem_now = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            logger.info(f"Memory usage before reconstruction: {mem_now:.2f} MB")
            
            logger.info("Reconstructing flood inundation map using SDR algorithm")
            depth_map = reconstruct_flood_inundation_map(self.demfile, self.target_xy_grid, 
                                                        self.rls_shp_file, self.sdr_thalwegs_file, 
                                                        pred_depths_list)
            
            mem_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            logger.info(f"Memory usage after reconstruction: {mem_after:.2f} MB")
            end = time.process_time()
            cpu_time = end - start
            memory_allocation = mem_after - mem_now
            
            srr_profile = {
                'max_cpu_memory': memory_allocation,
                'max_cuda_memory': 0,
                'total_cpu_time': cpu_time,
                'total_gpu_time': 0,
                'total_cpu_memory': 0,  # This can be calculated if needed
                'total_gpu_memory': 0,  # This can be calculated if needed
            }
            
            logger.info(f"Profile {srr_profile}")
            
        else:
            depth_map = reconstruct_flood_inundation_map(self.demfile, self.target_xy_grid, 
                                                     self.rls_shp_file, self.sdr_thalwegs_file, 
                                                     pred_depths_list)
        
        depth_map = torch.from_numpy(depth_map).float().to(self.device)
        ref_map = self.data_manager.get_batch(test_idx)
    
        logger.info(f"Depth map shape: {depth_map.shape}")
        logger.info(f"Reference map shape: {ref_map.shape}")       

        loss = self.loss_fn(depth_map, ref_map)
        nse = self.nse_fn(ref_map, depth_map)
        mRMSE = self.mRMSE_fn(depth_map, ref_map)
        
        
        logger.info(f"Map {map_i} MSE {loss.item()} NSE {nse} mRMSE {mRMSE} ")
        
        return {
            'map_i': map_i,
            'mse': loss.item(),
            'nse': nse,
            'mRMSE': mRMSE,
            'srr_profile': srr_profile
        }

    def reconstruct(self):
        logger.info("Running reconstruction")
        test_idxs = self.data_manager.reconstruction_test_idxs
        logger.info(f"Testing the recosntruction for : {test_idxs.shape} maps")
    
        start_time = time.time()
        
        logger.info("Predicting depths at representative locations using CNN models")
        pred_depths, pred_time, lstm_profile = self.predict_water_depths()
        logger.info(f"Time taken for RL depth prediction: {pred_time}")
        
        # Process first map with profiling
        first_result = self.process_single_map(0, test_idxs[0], pred_depths[0], profile_memory=True)
        srr_profile = first_result['srr_profile']
        
        # Prepare tasks for parallel processing (skip first map as it's already processed)
        tasks = []
        for map_i in range(1, len(test_idxs)):
            tasks.append((map_i, test_idxs[map_i], pred_depths[map_i]))
        
        results = [first_result]  # Start with first result
        
        # Process remaining maps in parallel
        if tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                # Submit all reconstruction tasks
                future_to_map = {
                    executor.submit(self.process_single_map, map_i, test_idx, pred_depths_map, False): map_i
                    for map_i, test_idx, pred_depths_map in tasks
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_map):
                    map_i = future_to_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed reconstruction for map {map_i}")
                    except Exception as exc:
                        logger.error(f"Map {map_i} generated an exception: {exc}")
        
        # Aggregate results
        overall_mse = sum(result['mse'] for result in results)
        overall_nse = sum(result['nse'] for result in results)
        overall_mRMSE = sum(result['mRMSE'] for result in results)
                
        end_time = time.time()
        
        mse = overall_mse / test_idxs.shape[0]
        rmse = np.sqrt(mse)
        mRMSE = overall_mRMSE / test_idxs.shape[0]
        nse = overall_nse / test_idxs.shape[0]
        reconstruction_time = end_time - start_time
        
        logger.info(f"Overall MSE: {mse}, RMSE: {rmse}, NSE: {nse} mRMSE: {mRMSE}")
        
        # Calculate CPU time used (this variable needs to be defined or removed)
        
        #Get max memory usage from profiling
        pred_memory_usage = {
            "max_cuda_memory": max(lstm_profile['max_cuda_memory'] * 2, srr_profile['max_cuda_memory']) if lstm_profile and srr_profile else 0,
            "max_cpu_memory": max(lstm_profile['max_cpu_memory'] * 2, srr_profile['max_cpu_memory']) if lstm_profile and srr_profile else 0, 
            "total_cpu_time": lstm_profile['total_cpu_time'] + srr_profile['total_cpu_time'] if lstm_profile and srr_profile else 0,
            "total_gpu_time": lstm_profile['total_gpu_time'] + srr_profile['total_gpu_time'] if lstm_profile and srr_profile else 0,
            "total_cpu_memory": 0,  
            "total_gpu_memory": 0  
        }
        
        total_flops = self.lstm_metrics['total_flops'] + srr_profile['flops'] if srr_profile else 0
        
        logger.info(f"Max CUDA Memory Usage: {pred_memory_usage['max_cuda_memory']} MB")
        logger.info(f"Max CPU Memory Usage: {pred_memory_usage['max_cpu_memory']} MB")
        logger.info(f"Total FLOPs: {format_flops(total_flops)}")
        
        metrics  = {
            'mse': mse,
            'rmse': rmse,
            'mRMSE': mRMSE,
            'nse': nse,
            'pred_time': reconstruction_time,
            'flops': total_flops,
            'pred_memory_usage':pred_memory_usage,
        }
        return metrics




