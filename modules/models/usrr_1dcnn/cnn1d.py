from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.utils.run_util import check_device
from modules.datamanager.point.sequential_loader_1dcnn import CNNSequentialDataManager
from modules.lib.constants import USRR_1DCNN_V1

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
    def __init__(self, model_structure, seq_h):
        super(CNN1DSequential, self).__init__()
        convo_1_kernel = 4
        pool_1_kernel = 3
        
        self.convo_1 = nn.Conv1d(in_channels=model_structure[0], out_channels=model_structure[1], 
                                 kernel_size=convo_1_kernel)
        self.pooling_1 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        self.convo_2 = nn.Conv1d(in_channels=model_structure[1],out_channels=model_structure[1], 
                                 kernel_size=convo_1_kernel)
        self.pooling_2 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        self.dim_past_convo = lambda dim_in: int(np.ceil((dim_in - convo_1_kernel + 1)/pool_1_kernel))
        flattened_dim = self.dim_past_convo(self.dim_past_convo(seq_h)) * model_structure[1]
        self.flatten = nn.Flatten()
        self.hidden_1 = nn.Linear(flattened_dim, model_structure[-2])
        self.lyr_out = nn.Linear(model_structure[-2], model_structure[-1])
        self.lrelu = nn.LeakyReLU()
        

    def forward(self, x):
        
        x = self.convo_1(x.transpose(1, 2))
        x = self.pooling_1(x)
        x = torch.tanh(x)
        x = self.convo_2(x)
        x = self.pooling_2(x)
        x = self.lrelu(x)
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
        self.rl_group = config.args.get("rl_group")  # This parameter has no default value
        self.num_of_clusters = self.config.args.get("n_clusters", 100)
        self.map_sampling_dist = config.args.get("sampling_dist", 20)
        self.rl_group_size = 0
        self.num_of_features = 3
    
        # Sequence
        self.timestep = 1
        self.input_time_len_h = self.config.args.get("input_time_len_h", 12)
        # 4 timesteps (15min) per hour
        self.seq_h = self.input_time_len_h * 4 
        self.device = check_device()
        self.tuninig_mode = config.args.get('tuning_mode', True)

    def create_dataset(self):
        self.data_manager = CNNSequentialDataManager(self.config.batch_size, self.input_time_len_h, self.rl_group, self.timestep,
                                                 self.map_sampling_dist, self.num_of_clusters, fold=self.config.fold,
                                                 tuning_mode=self.tuninig_mode)
        self.rl_group_size = self.data_manager.rl_group_size
    
    def init_model(self):
        logger.info(f"Initializing CNN1D model")
        self.create_dataset()
        # model structure [input_dim, conv_out_dim, hidden-fc-layer-size output_dim]
        self.model_structure = [self.num_of_features, 32, 64, self.rl_group_size]
        self.model = CNN1DSequential(self.model_structure, self.seq_h).to(self.device)
        self.model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        logger.info(f"Model initialized")
        return True 
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
        
    def test_model(self):
        return super().test_model()
        
