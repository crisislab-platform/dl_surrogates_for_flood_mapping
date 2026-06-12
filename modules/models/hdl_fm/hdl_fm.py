import time

from modules.models.model_wrapper import Config, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import HDL_FM_V1
from modules.datamanager.hdlfm.hdl_fm_dm import HDLFMDataManager
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HDLFM_ModelWrapper")
model_name  = HDL_FM_V1

# class HDLFMModel(nn.Module):
    
#     def __init__(self, output_shape):
#         super(HDLFMModel, self).__init__()
#         # CNN layers
#         input_channels = 3
#         cnn1_channels = 13
#         cnn2_channels = 5
#         H = output_shape[0]
#         W = output_shape[1]

#         # Calculate dimensions after CNN layers with pooling
#         H_after_cnn = H // (4*3*2)  # Three max-pooling layers with stride 2
#         W_after_cnn = W // (4*3*2)  # Three max-pooling layers with stride 2
#         lstm_input_size = H_after_cnn * W_after_cnn  # Size after flattening
#         lstm_hidden_size = lstm_input_size
#         lstm_num_layers = 1
    
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channels, cnn1_channels, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=4, stride=4),
#             nn.Conv2d(cnn1_channels, cnn2_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=3),
#             nn.Conv2d(cnn2_channels, 1, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         #Flatten Layer
#         self.flatten = nn.Flatten()
        
#         #LSTM Layer
#         self.lstm = nn.LSTM(input_size=lstm_input_size, 
#                             hidden_size=lstm_hidden_size, 
#                             num_layers=lstm_num_layers, 
#                             batch_first=True)
        
#         #Linear Layer
#         self.linear = nn.Linear(lstm_hidden_size, H*W)
    
#     def forward(self, x):
#         # x shape: (batch_size, channels, height, width)
#         batch_size = x.size(0)
        
#         # CNN
#         x = self.cnn(x)
        
#         # Flatten
#         x = self.flatten(x)
    
#         #LSTM
#         x, _ = self.lstm(x.unsqueeze(1))  # Add sequence dimension

#         #Linear
#         x = self.linear(x)
#         return x

class GaugeEncoder(nn.Module):
    """Encodes each gauge scalar independently as its own token"""
    def __init__(self, temporal_features, d_model):
        self.temporal_features = temporal_features
        super().__init__()
        self.fc = nn.Linear(1, d_model)  # encode each gauge independently

    def forward(self, features):
        # features: (B, N_gauges)
        x = features.unsqueeze(-1)   # (B, N_gauges, 1)
        return self.fc(x)            # (B, N_gauges, d_model)
    
class HDLFMModel(nn.Module):
    def __init__(self, output_shape, input_channels=3, temporal_features=3):
        super(HDLFMModel, self).__init__()
        H, W = output_shape  # 2448, 2416
        self.temporal_features = temporal_features
        cnn1_channels = 13
        cnn2_channels = 5
        feature_embed_dim = 64
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn1_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn1_channels, cnn2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn2_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        spatial_size = (H // 8) * (W // 8)

        # # Project spatial features for  d_model for attention
        # self.spatial_proj = nn.Linear(1, feature_embed_dim)  # Project CNN output to feature_embed_dim

        # self.gauge_encoder = GaugeEncoder(temporal_features=temporal_features, d_model=feature_embed_dim)

        # # Cross-attention: spatial features query gauge tokens
        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=feature_embed_dim,
        #     num_heads=4,
        #     batch_first=True
        # )
       
        # # Project back to scalar per spatial location
        # self.attn_proj = nn.Linear(feature_embed_dim, 1)
       

        lstm_input_size = spatial_size # After three max-pooling layers with stride 2
        lstm_hidden_size = lstm_input_size
        lstm_num_layers = 1
        
        # # Flatten layer
        self.flatten = nn.Flatten()
        
        # # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # Linear layer
        self.linear = nn.Linear(lstm_input_size, H*W)

    def forward(self, x):
        # temporal = x[:, :self.temporal_features, :, :]
        # spatial = x[:, self.temporal_features:, :, :]  
         
        # CNN on spatial features only
        x = self.cnn(x)                 # (B, 1, H//8, W//8)
        # B, _, h, w = spatial_out.shape

        # # Reshape for attention: (B, h*w, 1)
        # spatial_out = spatial_out.permute(0, 2, 3, 1).reshape(B, h * w, 1)

        # # Project to d_model
        # queries = self.spatial_proj(spatial_out)        # (B, h*w, d_model)

        # # Encode gauge tokens
        # temporal = temporal.mean(dim=[2,3])              # (B, temporal_features)
        # gauge_tokens = self.gauge_encoder(temporal) 

        # # Cross-attention: spatial locations attend to gauges
        # attn_out, _ = self.cross_attn(
        #     query=queries,                              # (B, h*w, d_model)
        #     key=gauge_tokens,                           # (B, N_gauges, d_model)
        #     value=gauge_tokens                          # (B, N_gauges, d_model)
        # )                                               # (B, h*w, d_model)

        # # Project back to scalar + residual
        # attn_out = self.attn_proj(attn_out)             # (B, h*w, 1)
        # x = spatial_out + attn_out                      # residual (B, h*w, 1)
        # x = x.squeeze(-1)                               # (B, h*w)
        x = self.flatten(x)
    
        # LSTM
        x, _ = self.lstm(x.unsqueeze(1))               # (B, 1, h*w)

        # Linear
        x = self.linear(x)                             # (B, 1, H*W)
        return x
                            
class HDLFMModelWrapper(ModelWrapper):
    def __init__(self, config: Config):
        super().__init__(config)
        self.autoregressive_model = True
        self.model_name = model_name
        self.device = check_device()

    def create_dataset(self):
        logger.info("Creating HDLFM dataset")
        self.data_manager = HDLFMDataManager(self.config)
        self.input_features = self.data_manager.temporal_features
        self.outputs = self.data_manager.output_shape
        logger.info(f"Features: {self.input_features}, Outputs: {self.outputs}")
    
    def init_model(self):
        self.create_dataset()
        self.model = HDLFMModel(self.outputs, self.data_manager.input_channels, self.data_manager.temporal_features).to(self.device)
        
        # Exact criterion from the oringal study
        self.loss_fn= nn.SmoothL1Loss(beta=0.75)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.75,
            patience=5,
            mode = 'min', 
            threshold=0.01 if self.config.tuning_mode else 0.001
        )
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        return True
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    
    # def train_model(self, run_dir: str, tuning_mode = False):
    #     self.setup_file_logging(os.path.join(run_dir, "training.log"))
    #     torch.cuda.empty_cache()
    #     logger.info(f"Saving model training history {self.config.save_model}")
    #     device = self.device
    #     history = {
    #         "loss": [],
    #         "val_loss": [],
    #         "train_time": None
    #     }

    #     start_time = time.time()
    #     self.best_val_loss = float("inf")
    #     self.epochs_no_improvement = 0
    #     self.best_epoch = 0
    #     total_steps = 0
    #     for epoch in range(self.config.epochs):
    #         epoch_loss = 0
    #         valid_batches = 0
    #         self.model.train()
    #         tqdm_loader = tqdm(self.data_manager.train_data_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
    #         for input_batch, output_batch, indices in tqdm_loader:
    #             input_batch = input_batch.to(device)
    #             output_batch = output_batch.to(device)
                
    #             #check if a batch has five dimentions.
    #             if len(input_batch.shape) == 5:
    #                 B, T, C, H, W = input_batch.shape
    #                 input_batch = input_batch.view(B * T, C, H, W)
    #             if len(output_batch.shape) == 5:
    #                 B, T, C, H, W = output_batch.shape
    #                 output_batch = output_batch.view(B * T, C, H, W)
                
    #             if input_batch is None or output_batch is None:
    #                 logger.warning(f"Error getting batch skipping")
    #                 continue
            
    #             pred = self.model(input_batch)
    #             if pred.shape != output_batch.shape:
    #                 output_batch = output_batch.squeeze(1)
    #                 B, H, W = output_batch.shape
    #                 pred = pred.view(B, H, W)
                    
    #             #compute the reconstruction loss
    #             #each batch should contain the complete set of tiles for a given time step. 
    #             #If not, we need to mask the loss to only compute over the tiles that are present in the batch. We can use the indices to determine which tiles are present in the batch and create a mask accordingly.
    #             #First group the indices by time step
    #             # recon_batch_loss = torch.tensor(0.0, device=device)

    #             # if total_steps > -1:
    #             #     event_time_steps = {}
    #             #     for idx, batch_idx in enumerate(indices):
    #             #         event_id = self.data_manager.find_event_id(batch_idx.item())
    #             #         time_step, _ = self.data_manager.find_local_indices(event_id, batch_idx.item())
    #             #         event_time_steps.setdefault((event_id, time_step), []).append(pred[idx: idx + self.data_manager.tiles_per_index].clone().detach())

    #             #     n_recon = 0
    #             #     for (event_id, time_step), timestep_preds in event_time_steps.items():
    #             #         timestep_preds_tensor = torch.concat(timestep_preds, dim=0)
    #             #         if timestep_preds_tensor.shape[0] < self.data_manager.tiles_per_index * self.data_manager.indices_per_timestep:
    #             #             continue

    #             #         recon_flood_map = self.data_manager.map_sampler.reconstruct_full_map(timestep_preds_tensor)
    #             #         ref_out = self.data_manager.get_flood_map(event_id, time_step).to(device)
    #             #         recon_batch_loss += self.loss_fn(recon_flood_map, ref_out)
    #             #         n_recon += 1

    #             #     if n_recon > 0:
    #             #         recon_batch_loss = recon_batch_loss / n_recon

    #             batch_loss = self.loss_fn(pred, output_batch) #+ recon_batch_loss
    #             self.optimizer.zero_grad()
    #             batch_loss.backward()
    #             self.optimizer.step()
    #             epoch_loss += batch_loss.item()
    #             total_steps += 1
    #             tqdm_loader.set_postfix({"Batch Loss": batch_loss.item()})
    #             valid_batches += 1
                
    #         # Divide by actual number of valid batches processed
    #         epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
    #         history["loss"].append(epoch_loss)
            
    #         if self.config.tuning_mode:
    #             validation_loss = self.validation(epoch)
    #             history["val_loss"].append(validation_loss)
    #             logger.info(f"Epoch {epoch + 1} loss: {epoch_loss}  validation loss: {validation_loss} ")

    #             if self.scheduler:
    #                 if isinstance(self.scheduler, ReduceLROnPlateau):
    #                     self.scheduler.step(validation_loss)
    #                 else:
    #                     self.scheduler.step()
                            
    #             if validation_loss < self.best_val_loss:
    #                 self.best_val_loss = validation_loss
    #                 self.epochs_no_improvement = 0
    #                 self.best_epoch = epoch
    #             else:
    #                 self.epochs_no_improvement += 1
    #                 if self.epochs_no_improvement >= self.config.patience:
    #                     logger.info(f"Early stopping at epoch {epoch + 1}")
    #                     logger.info(f"Best validation loss: {self.best_val_loss} at epoch {self.best_epoch}")
    #                     break
    #         else:
    #             logger.info(f"Epoch {epoch + 1} loss: {epoch_loss} ")
    #             if self.scheduler:
    #                 if isinstance(self.scheduler, ReduceLROnPlateau):
    #                     self.scheduler.step(epoch_loss)
    #                 else:
    #                     self.scheduler.step()
        
            
    #     # Add required metrics to history
    #     if tuning_mode:
    #         history["best_val_loss"] = self.best_val_loss
    #         history["best_epoch"] = self.best_epoch
            
    #     # Create hyperparameters dictionary
    #     hyperparameters = self.create_hyperparameters_dict()
    #     history["hyperparameters"] = json.dumps(hyperparameters)
                
    #     end_time = time.time()
    #     train_time = end_time - start_time
    #     logger.info(f"Training time: {train_time}")
    #     history["train_time"] = train_time
    
    #     # Save the model state
    #     model_file = None
    #     if self.config.save_model:
    #         logger.info(f"Saving model state to {run_dir}")
    #         model_state = self.model.state_dict().copy()
    #         model_file = self.save_model_checkpoint(model_state, self.config)
    #     return history, train_time, model_file
    