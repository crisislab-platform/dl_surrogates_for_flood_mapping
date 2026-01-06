from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import HDL_FM_V1
from modules.datamanager.raster.raster_loader_hdl_fm import HDLFMRasterDataManager
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map
from modules.lib.constants import OUTPUT_DIR, SIMULATION_DATA_DIR, RUN_DIR
import time
import os
import numpy as np
import rasterio
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HDLFM_ModelWrapper")
model_name  = HDL_FM_V1

class HDLFMModel(nn.Module):
    
    def __init__(self):
        super(HDLFMModel, self).__init__()
        # CNN layers
        input_channels = 3
        cnn1_channels = 13
        cnn2_channels = 5
        H = 611
        W = 951
        
        # Calculate dimensions after CNN layers with pooling
        H_after_cnn = H // (4*3*2)  # Three max-pooling layers with stride 2
        W_after_cnn = W // (4*3*2)  # Three max-pooling layers with stride 2
        lstm_input_size = H_after_cnn * W_after_cnn  # Size after flattening
        lstm_hidden_size = lstm_input_size
        lstm_num_layers = 1
    
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn1_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(cnn1_channels, cnn2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(cnn2_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #Flatten Layer
        self.flatten = nn.Flatten()
        
        #LSTM Layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        #Linear Layer
        self.linear = nn.Linear(lstm_hidden_size, H*W)
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.size(0)
        
        # CNN
        x = self.cnn(x)
        
        # Flatten
        x = self.flatten(x)
    
        #LSTM
        x, _ = self.lstm(x.unsqueeze(1))  # Add sequence dimension

        #Linear
        x = self.linear(x)
        return x

class HDLFMModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        self.device = check_device()
        self.lag = config.lag
        self.steps = 1
        self.features = self.lag * 3
        self.outputs = 581061
        self.validation_event = self.config.fold  + 1
        self.tuninig_mode = config.args.get('tuning_mode', False)

    def create_dataset(self):
        logger.info("Creating HDLFM dataset")
        self.data_manager = HDLFMRasterDataManager(
            lag=self.config.lag, 
            batch_size=self.config.batch_size, 
            validation_event=self.validation_event, 
            tuning_mode=self.tuninig_mode
        )
        self.features = self.data_manager.features
        self.outputs = self.data_manager.outputs
    
    def init_model(self):
        self.device = check_device()
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        self.create_dataset()
        self.model = HDLFMModel().to(self.device).float()
        self.loss_fn= nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return True
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        logger.info("Testing model")
        self.model.eval()
        predictions =  []
        ground_truth = []
        start_time = time.time()
        
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
            with torch.no_grad():
                total_loss = 0.0
                total_mRMSE = 0.0
                total_nse = 0.0
                all_tp = 0
                all_tn = 0
                all_fp = 0
                all_fn = 0
                for index in self.data_manager.test_index:
                    input_data, ref_out = self.data_manager.get_test_batch(index)
                    pred = self.model(input_data)
                    # If pred < 0.3 then set to 0
                    pred = torch.where(pred < 0.3, torch.tensor(0.0).to(self.device), pred)
                    
                    loss = self.loss_fn(pred, ref_out)
                    
                    # Calculate MSE
                    batch_mse = loss.item()
                    total_loss += batch_mse
                
                    # Calculate mRMSE for wet cells
                    batch_mRMSE = self.mRMSE_fn(pred, ref_out)
                    total_mRMSE += batch_mRMSE
                    
                    # confusion matrix components
                     #calculate confusion matrix at 0.3m threshold
                    threshold = 0.3
                    pred_binary = (pred > threshold).float()
                    ref_binary = (ref_out > threshold).float()
                    tp = ((pred_binary == 1) & (ref_binary == 1)).sum().item()
                    tn = ((pred_binary == 0) & (ref_binary == 0)).sum().item()
                    fp = ((pred_binary == 1) & (ref_binary == 0)).sum().item()
                    fn = ((pred_binary == 0) & (ref_binary == 1)).sum().item()
                    
                    all_tp += tp
                    all_tn += tn
                    all_fp += fp
                    all_fn += fn

                    # Calculate NSE
                    observed = ref_out
                    predicted = pred
                    batch_nse = self.nse_fn(observed, predicted)
                    total_nse += batch_nse
                    predictions.append(pred)
                    ground_truth.append(ref_out)
                    logger.info(f"Test index: {index}, Loss: {batch_mse}, mRMSE: {batch_mRMSE}, NSE: {batch_nse}")
                    prof.step()
                    
                # Average the losses
                mse = total_loss / len(self.data_manager.test_index)
                mRMSE = total_mRMSE / len(self.data_manager.test_index)
                nse = total_nse / len(self.data_manager.test_index)
                rmse = np.sqrt(mse)
                
                # Calculate Hit Ratio and Critical Success Index (CSI)
                hit_ratio = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn) if (all_tp + all_tn + all_fp + all_fn) > 0 else 0
                csi = all_tp / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
                # F2 Score
                f2_score = (all_tp - all_fn) / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
                # F3 Score
                f3_score = (all_tp - all_fp) / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0
               
        
        end_time = time.time()
        pred_time = end_time - start_time
        logger.info(f"Validation prediction_time:{pred_time} loss MSE: {mse} RMSE: {rmse}  NSE: {nse} mRMSE: {mRMSE} Hit Ratio: {hit_ratio} CSI: {csi} F2 Score: {f2_score} F3 Score: {f3_score}")
        
        predictions = torch.cat(predictions, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        # Save predictions as raster
        flops = self.calculate_flops()
        self.save_predictions_all(predictions)
                
        # Save predictions at points of interest
        poi_path = os.path.join(OUTPUT_DIR, "points_of_interest.csv")
        if os.path.exists(poi_path):
            self.save_predictions_at_points(predictions, ground_truth, poi_path)
        else:
            logger.warning(f"Points of interest file not found at {poi_path}")
                
        analysis_results = profiler_analysis(prof.key_averages())
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "nse": nse,
            "mRMSE": mRMSE,
            "pred_time": pred_time,
            "flops": flops,
            "pred_memory_usage": analysis_results, 
            "hit_rate": hit_ratio,
            "csi": csi,
            "f2_score": f2_score,
            "f3_score": f3_score
        }
        
        logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        return metrics
    
