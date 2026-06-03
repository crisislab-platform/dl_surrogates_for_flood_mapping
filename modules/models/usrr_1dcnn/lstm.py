from modules.models.model_wrapper import ModelWrapper, Config
from modules.utils.run_util import check_device
from modules.datamanager.point.sequential_loader_1dcnn import CNNSequentialDataManager
from modules.lib.constants import USRR_1DCNN_V1, RUN_DIR

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import logging
import time
import os
import json
import pandas as pd

model_name = USRR_1DCNN_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTM_USRR_ModelWrapper")

class LSTM(nn.Module):
    """
    Simpler LSTM model without attention for baseline comparison
    """
    def __init__(self, model_structure, num_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        
        self.input_size = model_structure[0]
        self.hidden_size = model_structure[1]
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, model_structure[2])
        self.bn1 = nn.BatchNorm1d(model_structure[2])
        self.dropout1 = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(model_structure[2], model_structure[-1])
        self.relu = nn.ReLU()

        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.bn1(x)
        x = self.relu(x)
        output = self.output_layer(x)
        return output

class LSTMModelWrapper(ModelWrapper):
    
    def __init__(self, config: Config):
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
        self.num_layers = self.config.args.get("lstm_layers", 2)
        self.tuninig_mode = config.args.get('tuning_mode', True)
        self.hidden_size = self.config.args.get("hidden_size", 64)
        self.fc_layer_size = self.config.args.get("fc_layer_size", 128)
        self.device = check_device()
        logger.info(f"Initializing LSTM model with hidden_size: {self.hidden_size} fc_layer_size: {self.fc_layer_size}")
        
    def create_dataset(self):
        self.data_manager = CNNSequentialDataManager(self.config.batch_size, self.input_time_len_h, self.rl_group,
                                                 self.map_sampling_dist, self.num_of_clusters, self.config.fold,
                                                 self.tuninig_mode)
        self.rl_group_size = self.data_manager.rl_group_size
    
    def init_model(self):
        logger.info(f"Initializing LSTM model")
        
        # Set deterministic seeds for model initialization
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        self.create_dataset()
        # model structure [input_dim, hidden_size, fc_layer_size, output_dim]
        self.model_structure = [self.num_of_features, self.hidden_size, self.fc_layer_size, self.rl_group_size]
        
        self.model = LSTM(self.model_structure,
                                  num_layers=self.num_layers,
                                  dropout=self.config.dropout).to(self.device)
        
        self.model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        
        logger.info(f"LSTM model initialized")
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
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
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
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
            }, model_file) 
            return model_file      
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None
        
    def save_predictions(self, pred):
        pass

    def save_predictions_all(self, pred):
        pass

    def save_predictions_at_points(self, pred, ref_out, poi_path):
        pass