from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.utils.run_util import check_device
from modules.dataloader.sequential_1dcnn.cnn_datamanager import CNNDataManager
from modules.lib.constants import USRR_1DCNN_V1, RUN_DIR
from torch.utils.flop_counter import FlopCounterMode
from torch.profiler import profile, ProfilerActivity

import numpy as np
import torch
import torch.nn as nn
import logging
import time
import os
import pandas as pd

model_name = USRR_1DCNN_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNN1DModel")

class CNN1DModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        
        # RL variables
        self.rl_group= config.args.get("rl_group")
        self.num_of_clusters = self.config.args.get("n_clusters", 100)
        self.map_sampling_dist = config.args.get("sampling_dist", 20)
        self.rl_group_size = 0
        self.num_of_features = 3
        # model structure [input_dim, conv1_out_dim, conv2_out_dim, hidden, output_dim]
        self.model_structure = [self.num_of_features, 32, 32, 128, self.rl_group_size]
    
        # Sequence variables(revisit)
        self.timestep = 1 #(15 min)
        self.time_lag_h = config.lag
        self.input_time_len_h = self.config.args.get("input_time_len_h", 12) # 12 hours default
        self.seq_h = self.input_time_len_h * 4 #4 timesteps (15min) per hour

    def create_dataset(self):
        self.data_manager = CNNDataManager(self.config.batch_size, self.input_time_len_h, 
                                                 self.time_lag_h, self.rl_group, self.timestep,
                                                 self.map_sampling_dist, self.num_of_clusters)
        self.rl_group_size = self.data_manager.rl_group_size
        model_structure = self.model_structure
        model_structure[-1] = self.rl_group_size
        self.model_structure = model_structure
        
    def init_model(self):
        logger.info(f"Initializing CNN1D model")
        self.create_dataset()
        device = check_device()
        self.model = ConvoModel2RLs(self.model_structure, self.seq_h).to(device)
        # self.model = CNNModel(self.seq_h, self.num_of_features, self.rl_group_size).to(device)
        self.model.float()
        self.loss_function = nn.MSELoss()
        self.eval_loss_function = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        logger.info(f"Model initialized")
        return True 
        
    def train(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        train_loader  = self.data_manager.train_loader
        val_loader = self.data_manager.val_loader
        device = check_device()
        
        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None

        history = {
            "loss": [],
            "eval_loss": [],
            "val_loss": [],
            "eval_val_loss": [],
            "train_time": None
        }
        
        losses = []
        eval_losses = []
        val_losses = []
        eval_val_losses = []
        best_model_state = None
        best_optimizer_state = None
        train_start = time.time()
        
        # Add profiler wrapper
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir)) as prof:
            
            for epoch in range(self.config.epochs):
                batch_losses = 0
                batch_eval_losses = 0
                batch_val_losses = 0
                batch_eval_val_losses = 0
                
                for input_in_batch, output_in_batch  in train_loader:
                    input_in_batch = input_in_batch.to(device)
                    output_in_batch = output_in_batch.to(device)
                    self.model.train()
                    self.optimizer.zero_grad()
                    pred = self.model(input_in_batch.float())
                    output_in_batch = output_in_batch.float()
                    loss = self.loss_function(pred, output_in_batch)
                    batch_losses += loss.item()
                    eval_loss = self.eval_loss_function(pred, output_in_batch).detach().cpu().numpy()
                    batch_eval_losses += eval_loss
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"Batch train loss: {loss.item()} eval loss: {eval_loss}")
            
                with torch.no_grad():
                    for input_in_batch, output_in_batch in val_loader:
                        input_in_batch = input_in_batch.to(device)
                        output_in_batch = output_in_batch.to(device)
                        self.model.eval()
                        pred_val = self.model(input_in_batch.float())
                        output_in_batch = output_in_batch.float()
                        val_loss = self.loss_function(pred_val, output_in_batch)
                        eval_val_loss = self.eval_loss_function(pred_val, output_in_batch).detach().cpu().numpy()
                        batch_val_losses += val_loss.item()
                        batch_eval_val_losses += eval_val_loss
                        logger.info(f"Batch validation loss: {val_loss.item()} eval loss: {eval_loss}")
                
                loss = batch_eval_losses/len(train_loader)
                val_loss = batch_val_losses/len(val_loader)
                eval_loss =  batch_eval_losses/len(train_loader)
                eval_val_loss = batch_eval_val_losses/len(val_loader)
                
                history["loss"].append(loss)
                history["eval_loss"].append(val_loss)
                history["val_loss"].append(eval_loss)
                history["eval_val_loss"].append(eval_val_loss)
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Train Loss: {loss}, Validation Loss: {val_loss}, Eval Train Loss: {eval_loss}, Eval Validation Loss: {eval_val_loss}")
                
                #Add early stopping logic here
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = self.model.state_dict()
                    best_optimizer_state = self.optimizer.state_dict()
                    losses.append(loss)
                    eval_losses.append(eval_loss)
                    val_losses.append(val_loss)
                    eval_val_losses.append(eval_val_loss)
                
                
                elif (epoch - best_epoch > self.config.patience) or (epoch == self.config.epochs - 1):
                    train_end = time.time()
                    logger.info(f"Early stopping at epoch {epoch+1} best epoch {best_epoch+1} with valdation loss: {best_val_loss}") 
                    break

        train_end = time.time()
        train_time = train_end - train_start
        history["train_time"] = train_time
        logger.info(f"Training completed in {train_end - train_start} seconds")
        
        key_averages = prof.key_averages()
        analysis_results = super().profiler_analysis(key_averages)
        logger.info(f"Memory profiling results: {key_averages.table(sort_by='cuda_memory_usage', row_limit=10)}")
        logger.info(f"Profiler analysis results: {analysis_results}")
        history['memory'] = analysis_results
        model_file = self.save_model_checkpoint(self.config.run_id, run_dir, best_model_state, best_optimizer_state)
        logger.info(f"Model saved to {model_file}")
        return history, train_time, model_file
        
    def validate_model(self):
        test_data_manager =   CNNDataManager(self.config.batch_size, self.input_time_len_h, 
                                                 self.time_lag_h, self.rl_group, self.timestep,
                                                 self.map_sampling_dist, self.num_of_clusters, test_mode=True)
        test_inputs = test_data_manager.test_inputs[0]
        test_ref = test_data_manager.test_inputs[1]
        
        # Add profiler wrapper for validation
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
            with torch.no_grad():
                self.model.eval()
                device = check_device()
                test_inputs = test_inputs.to(device)
                start_time = time.time()
                y_hat = self.model(test_inputs.float())
                end_time = time.time()
                prediction_time = end_time - start_time
                logger.info(f"Prediction completed in {prediction_time} seconds")
                mse = self.loss_function(y_hat, test_ref.to(device).float()).item()
                rmse = np.sqrt(mse)
                nse = self.nse_fn(y_hat, test_ref.to(device).float())
                flop_input = test_inputs[0:1]
                flops = self.calculate_flops(flop_input)
                logger.info(f"MSE: {mse}, RMSE: {rmse} NSE: {nse} FLOPS: {flops}")
                analysis_results = super().profiler_analysis(prof.key_averages())
                metrics = { 
                    "mse": mse,
                    "rmse": rmse,
                    "nse": nse,
                    "pred_time": prediction_time,
                    "flops": flops,
                    "wet_rmse": None,
                    "wet_acc": None,
                    "dry_rmse": None,
                    "dry_acc": None,
                    "pred_memory_usage": analysis_results
                }             
        # Log memory usage during validation
        logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        return metrics
    
    def calculate_flops(self, input):
        with FlopCounterMode(self.model) as counter:
            # _ = self.model(input, None, None, False)
            _ = self.model(input)
        flops = counter.get_total_flops()
        flops_str = self.format_flops(flops)
        logger.info(f"Model FLOPS: {flops_str}")
        return flops
    
    def save_model_checkpoint(self, run_id, run_dir, bets_model_state, best_optimizer_state):
        model_file = os.path.join(run_dir, f"{self.config.model_name}_{self.rl_group}_model.pt")
        metadata_dir = os.path.join(RUN_DIR, self.config.model_name)
        metadata_file = os.path.join(metadata_dir, "run_metadata.csv")
        
        try:
            torch.save({
                'model_state_dict': bets_model_state,
                'optimizer_state_dict': best_optimizer_state,
                'sampling_dist': self.map_sampling_dist,
                'num_of_clusters': self.num_of_clusters,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'model_structure': self.model_structure,
                'num_epochs': self.config.epochs,
                'seq_h': self.seq_h,
                'run_id': run_id,
            }, model_file)
            logger.info(f"Model saved to: {model_file}")
            
            # Ensure the directory for metadata exists
            os.makedirs(metadata_dir, exist_ok=True)
            
            # Check if the metadata file exists and is not empty
            if os.path.exists(metadata_file):
                logger.info(f"Metadata file exists: {metadata_file}")
                run_metadata = pd.read_csv(metadata_file)
            else:
                logger.info(f"Creating new metadata file: {metadata_file}")
                run_metadata = pd.DataFrame(columns=["run_id", "sampling_dist", "cluster_size", "model_name", "rl_group", "model_file"])
            
            # Using pandas concat instead of deprecated append
            new_row = pd.DataFrame({
                "run_id": [run_id],
                "sampling_dist": [self.map_sampling_dist],
                "cluster_size": [self.num_of_clusters],
                "model_name": [self.model_name],
                "rl_group": [self.rl_group],
                "model_file": [model_file]
            })
            run_metadata = pd.concat([run_metadata, new_row], ignore_index=True)
            
            run_metadata.to_csv(metadata_file, index=False)  
            
            logger.info(f"Run metadata saved to {metadata_file}")
            return model_file
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
        return None
        
class ConvoModel2RLs(nn.Module):
    
    def __init__(self, model_structure, seq_h):
        super(ConvoModel2RLs, self).__init__()
        convo_1_kernel = 4
        pool_1_kernel = 3
        self.convo_1 = nn.Conv1d(in_channels=model_structure[0], 
                                 out_channels=model_structure[1], 
                                 kernel_size=convo_1_kernel)
        self.pooling_1 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        self.batch_norm_1 = nn.BatchNorm1d(model_structure[1])
        self.convo_2 = nn.Conv1d(in_channels=model_structure[1],
                                 out_channels=model_structure[2], 
                                 kernel_size=convo_1_kernel)
        self.pooling_2 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        self.batch_norm_2 = nn.BatchNorm1d(model_structure[2])
        self.dim_past_convo = lambda dim_in: int(np.ceil((dim_in - convo_1_kernel + 1)/pool_1_kernel))
        flatten_dim = self.dim_past_convo(self.dim_past_convo(seq_h)) * model_structure[2]
        self.flatten = nn.Flatten()
        self.hidden_1 = nn.Linear(flatten_dim, model_structure[-2])
        self.lyr_out = nn.Linear(model_structure[-2], model_structure[-1])
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.convo_1(x.transpose(1, 2))
        x = self.pooling_1(x)
        x = self.batch_norm_1(x)
        x = torch.tanh(x)
        x = self.convo_2(x)
        x = self.pooling_2(x)
        x = self.batch_norm_2(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.lyr_out(x).squeeze(1)
        return x
    
# class CNNModel(nn.Module):
    
#     def __init__(self, seq_h, features, outputs):
#         kernal_size = 3
#         conv1_out_dim = 32
#         conv2_out_dim = 64
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=features, out_channels=conv1_out_dim, kernel_size=kernal_size, padding=1)
#         self.bn1 = nn.BatchNorm1d(conv1_out_dim)
#         self.conv2 = nn.Conv1d(in_channels=conv1_out_dim, out_channels=conv2_out_dim, kernel_size=kernal_size, padding=1)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(conv2_out_dim * (seq_h),128)
#         self.fc2 = nn.Linear(128, outputs)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = x.transpose(1, 2)
        
#         # First convolutional block
#         x = self.conv1(x)
#         x = self.relu(x)
        
#         # Second convolutional block
#         x = self.conv2(x)
#         x = self.relu(x)
        
#         # Fully connected layers
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x