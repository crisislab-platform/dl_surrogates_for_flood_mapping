from modules.lib.constants import SIMULATION_DATA_DIR, OUTPUT_DIR, RUN_DIR
from modules.models.srr_lstm.srr.sdr_algorithm_reco import reconstruct_flood_inundation_map
from modules.lib.constants import LSTM_SRR_V1
from modules.models.srr_lstm.srr_lstm import LSTMModel
import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SDRReconstructor")

class SDRReconstructor():
    def __init__(self):
        self.dem_asc_file = f"{SIMULATION_DATA_DIR}/Carlisle_5m.asc"
        self.max_inundation_timestep = {
            'event_id': 3,
            'timestep': 94
        }
        self.max_inundation_file = f"{SIMULATION_DATA_DIR}/Run{self.max_inundation_timestep['event_id']}-{(self.max_inundation_timestep['timestep']):04d}.wd"
        self.stopping_values_dem = [-555]
        self.reduction_result_dir = os.path.join(OUTPUT_DIR, "sdr_reduction_results")
        self.reco_result_dir = os.path.join(OUTPUT_DIR, "sdr_reconstruction_results")
        self.rls_shp_file = os.path.join(self.reduction_result_dir, "representative_locations.shp")
        self.sdr_thalwegs_file = os.path.join(self.reduction_result_dir, "sdr_results_thalwegs.shp")
        self.x_coords = None
        self.y_coords = None
        self.tidal_boundary_grids = None
        self.sealvl = None
        
        self.loss_fn = nn.MSELoss()
        self.mRMSE_fn = self.mRMSE_fn
        self.nse_fn = self.nse_fn
        
        
    def find_lstm_models(self):
        metrics_file = os.path.join(RUN_DIR, LSTM_SRR_V1, "final_training_metrics.csv")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metadata file {metrics_file} not found.")
        metrics_df = pd.read_csv(metrics_file)
    
        # group by  rl_group and get the latest model file by run_id
        latest_model_files = {}
        for rl_group in metrics_df['rl_group'].unique():
            group = metrics_df[metrics_df['rl_group'] == rl_group]
            latest_run_id = group['run_id'].sort_values(ascending=False).iloc[0]
            latest_model_file = group[group['run_id'] == latest_run_id]['model_file'].values[0]
            latest_model_files[rl_group] = latest_model_file      
        return latest_model_files

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
    
    def load_lstm_models(self, model_files):
        lstm_models = {}
        for rl_group, model_file in model_files.items():
            if not os.path.exists(model_file):
                logger.warning(f"Model file {model_file} for RL group {rl_group} does not exist.")
                continue
            lstm_models[rl_group] = self.load_cnn_model_from_file(model_file)
        return lstm_models
    
    def load_lstm_model_from_file(self, model_file):
        checkpoint  = torch.load(model_file)
        model_structue = checkpoint['model_structure']
        input_dim = model_structue['input_dim']
        hidden_dim = model_structue['hidden_dim']
        lstm_dim = model_structue['lstm_dim']
        output_dim = model_structue['output_dim']
        model = LSTMModel(input_dim, hidden_dim, lstm_dim, output_dim).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def predict_water_depths(self, lstm_models):
        pass
    
    def reconstruct(self):
        starttime = time()
        
        #Predict water depth using LSTM models
        lstm_model_files = self.find_lstm_models()
        lstm_models = self.load_lstm_models(lstm_model_files)
        lstm_predictions = self.predict_water_depths(lstm_models)
        
        #Reconsturct flood inundation map
        pred = reconstruct_flood_inundation_map(self.dem_asc_file, self.x_coords, self.y_coords, self.rls_shp_file, self.sdr_thalwegs_file, lstm_predictions, self.tidal_boundary_grids, self.sealvl)
        logger.info(f"SDR reconstruction completed in {time() - starttime:.2f} seconds")
        
        metrics = {}
        return metrics
        
def reconstruct_and_test():
    """
    This function orchestrates the reconstruction and testing of the SDR model.
    It initializes the SDRReconstructor, performs reconstruction, and returns metrics.
    """
    reconstructor = SDRReconstructor()
    
    