from modules.models.model_wrapper import ModelWrapper, ModelConfig
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
logger = logging.getLogger("CNN1D_USRR_ModelWrapper")

class CNN1DSequential(nn.Module):
    def __init__(self, model_structure, seq_h, convo_kernel=4, pool_kernel=3):
        super(CNN1DSequential, self).__init__()
        self.convo_1 = nn.Conv1d(in_channels=model_structure[0], out_channels=model_structure[1],
                                 kernel_size=convo_kernel, padding=1)
        
        self.pooling_1 = nn.MaxPool1d(pool_kernel, ceil_mode=True)
        
        self.convo_2 = nn.Conv1d(in_channels=model_structure[1],out_channels=model_structure[1],
                                 kernel_size=convo_kernel, padding=1)
        
        self.pooling_2 = nn.MaxPool1d(pool_kernel, ceil_mode=True)
        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(model_structure[1])
        self.bn2 = nn.BatchNorm1d(model_structure[1]*2)
        self.bn3 = nn.BatchNorm1d(model_structure[1]*2)
        self.dropout = nn.Dropout(0.1)
    
        with torch.no_grad():
            dummy = torch.zeros(1, model_structure[0], int(seq_h))
            flat_dim = self._forward_features(dummy).view(1, -1).size(1)
        
        self.flatten = nn.Flatten()
        self.hidden_1 = nn.Linear(flat_dim, model_structure[-2])
        self.hidden_2  = nn.Linear(model_structure[-2], model_structure[-2] * 2)
        self.hidden_3 = nn.Linear(model_structure[-2] * 2, model_structure[-2] * 4)
        self.lyr_out = nn.Linear(model_structure[-2], model_structure[-1])

    def _forward_features(self, x):
        x = self.dropout(self.relu(self.convo_1(x)))
        x = self.pooling_1(x)       
        x = self.dropout(self.relu(self.convo_2(x)))
        x = self.pooling_2(x)
        return x

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.hidden_1(x)))
        x = self.lyr_out(x)
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
        
        # Set deterministic seeds for model initialization
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        self.create_dataset()
        # model structure [input_dim, conv_out_dim, hidden-fc-layer-size output_dim]
        self.model_structure = [self.num_of_features,self.output_channel_size, self.fc_layer_size, self.rl_group_size]
        self.model = CNN1DSequential(self.model_structure, self.seq_h, 
                                    convo_kernel=self.convo_kernel, 
                                    pool_kernel=self.pool_kernel).to(self.device)
        # self.model = CNN1DSequential2(self.model_structure, 1, convo_kernel=self.convo_kernel).to(self.device)
        self.model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        
          
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',          # Reduce LR when val_loss stops decreasing
            factor=0.5,          # Multiply LR by this factor when reducing
            patience=5,          # Number of epochs with no improvement after which LR will be reduced
            verbose=True,        # Print message when LR is reduced
            min_lr=1e-5          # Lower bound on the learning rate
        )
        
        logger.info(f"Model initialized with learning rate scheduler (ReduceLROnPlateau)")
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

    def save_predictions_all(self, pred):
        pass

    def save_predictions_at_points(self, pred, ref_out, poi_path):
        pass