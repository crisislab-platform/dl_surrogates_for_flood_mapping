from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.utils.run_util import check_device
from modules.datamanager.point.sequential_loader_1dcnn import CNNSequentialDataManager
from modules.lib.constants import USRR_1DCNN_V1, RUN_DIR

import numpy as np
import torch
import torch.nn as nn
import logging
import time
import os
import pandas as pd

model_name = USRR_1DCNN_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNN1D_USRR_ModelWrapper")

class CNN1DSequential(nn.Module):
    def __init__(self, model_structure, seq_h, convo_kernel=4, pool_kernel=3):
        super(CNN1DSequential, self).__init__()
        self.convo_1 = nn.Conv1d(in_channels=model_structure[0], out_channels=model_structure[1], 
                                 kernel_size=convo_kernel, padding='same')
        self.pooling_1 = nn.MaxPool1d(pool_kernel, ceil_mode=True)
        
        self.convo_2 = nn.Conv1d(in_channels=model_structure[1],out_channels=model_structure[1], 
                                 kernel_size=convo_kernel, padding='same')
        
        self.pooling_2 = nn.MaxPool1d(pool_kernel, ceil_mode=True)
        
        self.dim_past_convo = lambda dim_in: int(np.ceil((dim_in)/pool_kernel))
        flattened_dim = self.dim_past_convo(self.dim_past_convo(seq_h)) * model_structure[1]
        
        # first_conv_out = int(((seq_h + 2*1 - convo_kernel) // stride) + 1)
        # second_conv_out = int(((first_conv_out + 2*1 - convo_kernel) // stride) + 1)
        # flattened_dim = int(second_conv_out * model_structure[1])
    
        self.flatten = nn.Flatten()
        self.hidden_1 = nn.Linear(flattened_dim, model_structure[-2])
        self.lyr_out = nn.Linear(model_structure[-2], model_structure[-1])
        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        
        self.batch_norm_1 = nn.BatchNorm1d(model_structure[1])
        self.batch_norm_2 = nn.BatchNorm1d(model_structure[1])
        
        self.dropout = nn.Dropout(0.2)

        
    def forward(self, x):
        x = self.convo_1(x.transpose(1, 2))
        x = torch.tanh(x)
        x = self.pooling_1(x)
        
        x = self.convo_2(x)
        x = self.lrelu(x)
        x = self.pooling_2(x)
        
        x = self.flatten(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.lyr_out(x).squeeze(1)
        return x
    
class CNN1DModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        
        # Representative location 
        self.rl_group = config.args.get("rl_group")
        self.num_of_clusters = self.config.args.get("n_clusters", 100)
        self.map_sampling_dist = config.args.get("sampling_dist", 20)
        self.rl_group_size = 0
        self.num_of_features = 3
    
        # Sequence
        self.input_time_len_h = self.config.args.get("input_time_len_h", 12)
        self.seq_h = self.input_time_len_h * 4 
        self.convo_kernel = self.config.args.get("conv_kernel", 4)
        self.pool_kernel = self.config.args.get("pool_kernel", 3)
        self.device = check_device()
        self.tuninig_mode = config.args.get('tuning_mode', True)
        self.output_channel_size = self.config.args.get("output_channel_size", 32)
        self.fc_layer_size = self.config.args.get("fc_layer_size", 64)
        logger.info(f"Initializing CNN1D model with output_seq_size: {self.output_channel_size} fc_layer_size: {self.fc_layer_size}")
        
    def create_dataset(self):
        self.data_manager = CNNSequentialDataManager(self.config.batch_size, self.input_time_len_h, self.rl_group,
                                                 self.map_sampling_dist, self.num_of_clusters, self.config.fold,
                                                 self.tuninig_mode)
        self.rl_group_size = self.data_manager.rl_group_size
    
    def init_model(self):
        logger.info(f"Initializing CNN1D model")
        self.create_dataset()
        # model structure [input_dim, conv_out_dim, hidden-fc-layer-size output_dim]
        self.model_structure = [self.num_of_features,self.output_channel_size, self.fc_layer_size, self.rl_group_size]
        self.model = CNN1DSequential(self.model_structure, self.seq_h, 
                                    convo_kernel=self.convo_kernel, 
                                    pool_kernel=self.pool_kernel).to(self.device)
        self.model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        logger.info(f"Model initialized")
        return True 
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
        
    def test_model(self):
        return super().test_model()
    
    def create_hyperparameters_dict(self):
        hyperparameters = {
            "learning_rate": self.config.learning_rate,
            'dropout': self.config.dropout,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "lag": self.config.lag,
            "horizon": self.config.horizon, 
            "sampling_dist": self.map_sampling_dist,
            "n_clusters": self.num_of_clusters,
            "rl_group": self.rl_group, 
            "input_time_len_h": self.input_time_len_h,
            "convo_kernel": self.convo_kernel,
            "pool_kernel": self.pool_kernel, 
            "output_channel_size": self.output_channel_size,
            "fc_layer_size": self.fc_layer_size
        } 
        return hyperparameters
    
    def save_model_checkpoint(self, run_id, run_dir, model_state, config):
        try:
            model_file = os.path.join(run_dir, f"{config.model_name}_{run_id}.pth")
            torch.save({
                'model_state_dict': model_state,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.epochs,
                'run_id': run_id,
                'model_structure': self.model_structure, 
                'rl_group': self.rl_group,
                'input_time_len_h': self.input_time_len_h,
                'convo_kernel': self.convo_kernel,
                'pool_kernel': self.pool_kernel,
            }, model_file) 
            return model_file      
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None
        
    def save_predictions(self, pred):
        pass
        # idx = 146
        # pred_max = pred.detach().cpu().numpy()[idx]
        # output_dir = os.path.join(self.config.run_dir, )
        # output_dir = os.path.join(RUN_DIR, model_name,"output_maps")  
        
        # os.makedirs(output_dir, exist_ok=True)
        # output_file = os.path.join(output_dir, f"1dcnn_predictions_{idx:04d}.csv")

        # # Get row and column coordinates from the data manager
        # coords = [(row, col) for row, col in self.data_manager.coords_to_cluster_rls]
        # rows = [coord[0] for coord in coords]
        # cols = [coord[1] for coord in coords]
        
        # # Create DataFrame with predictions and coordinates
        # prediction_map = pd.DataFrame({
        #     'row': rows,
        #     'col': cols,
        #     f'value_{self.rl_group}': pred_max.flatten()
        # })
        
        # if os.path.exists(output_file):
        #     df = pd.read_csv(output_file)
        #     # Merge on row and col if they exist in the file
        #     if 'row' in df.columns and 'col' in df.columns:
        #         # Keep only the columns from prediction_map that aren't row/col
        #         value_cols = [col for col in prediction_map.columns if col not in ['row', 'col']]
        #         # Merge the new predictions with existing data
        #         df = pd.merge(df, prediction_map[['row', 'col'] + value_cols], on=['row', 'col'], how='outer')
        #     else:
        #         # If existing file doesn't have coordinates, just use the new format
        #         df = prediction_map
        # else:
        #     df = prediction_map
            
        # df.to_csv(output_file, index=False)
        # logger.info(f"Prediction map saved to {output_file} with row/col coordinates")
# 0.055 m -  RMSE should be around this