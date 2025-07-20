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
        self.create_dataset()
        self.model = HDLFMModel().to(self.device)
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
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
            with torch.no_grad():
                total_loss = 0.0
                total_mRMSE = 0.0
                total_nse = 0.0
                for index in self.data_manager.test_index:
                    input_data, ref_out = self.data_manager.get_test_batch(index)
                    start_time = time.time()
                    pred = self.model(input_data)
                    # If pred < 0.3 then set to 0
                    pred = torch.where(pred < 0.3, torch.tensor(0.0).to(self.device), pred)
                    end_time = time.time()
                    loss = self.loss_fn(pred, ref_out)
                    
                    # Calculate MSE
                    batch_mse = loss.item()
                    total_loss += batch_mse
                
                    # Calculate mRMSE for wet cells
                    batch_mRMSE = self.mRMSE_fn(pred, ref_out)
                    total_mRMSE += batch_mRMSE

                    # Calculate NSE
                    observed = ref_out
                    predicted = pred
                    batch_nse = self.nse_fn(observed, predicted)
                    total_nse += batch_nse
                    if index == 146:
                        self.save_predictions(pred)
                    predictions.append(pred)
                    ground_truth.append(ref_out)
                    logger.info(f"Test index: {index}, Loss: {batch_mse}, mRMSE: {batch_mRMSE}, NSE: {batch_nse}")
                    prof.step()
                    
                # Average the losses
                mse = total_loss / len(self.data_manager.test_index)
                mRMSE = total_mRMSE / len(self.data_manager.test_index)
                nse = total_nse / len(self.data_manager.test_index)
                rmse = np.sqrt(mse)
                pred_time = end_time - start_time
                logger.info(f"Validation prediction_time:{pred_time} loss MSE: {mse} RMSE: {rmse}  NSE: {nse} mRMSE: {mRMSE}")
        
        predictions = torch.cat(predictions, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)
        
        flops = self.calculate_flops()
        # Save predictions at points of interest
        poi_path = os.path.join(OUTPUT_DIR, "points_of_interest.csv")
        if os.path.exists(poi_path):
            self.save_predictions_at_points(pred, ref_out, poi_path)
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
            "pred_memory_usage": analysis_results
        }
        
        logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        return metrics
    
    def save_predictions(self, pred):
        idx = 146
        pred_max = pred.detach().cpu().numpy()
        output_dir = os.path.join(self.config.run_dir, "output_maps")
        save_prediction_map(pred_max, output_dir, idx, self.config.model_name)
        
    def save_predictions_at_points(self, predictions, ground_truth, points_csv):
        """
        Save model predictions at specific points of interest to a CSV file
        
        Args:
            predictions: Model predictions tensor
            ground_truth: Ground truth tensor
            points_csv: Path to the CSV file containing points of interest
        """
        try:
            logger.info(f"Saving predictions at points of interest from {points_csv}")
            
            # Read points of interest
            poi_df = pd.read_csv(points_csv)
            
            # Convert tensors to numpy arrays for easier handling
            pred_np = predictions.detach().cpu().numpy()
            truth_np = ground_truth.detach().cpu().numpy()
            
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
                            'Model_Name': self.config.model_name,
                            'Run_ID': self.config.run_id
                        })
            
            # Create DataFrame and save to CSV
            results_df = pd.DataFrame(results)
            output_path = os.path.join(RUN_DIR, self.config.model_name,  f"predictions_at_points_final.csv")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved point predictions to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving predictions at points: {e}")