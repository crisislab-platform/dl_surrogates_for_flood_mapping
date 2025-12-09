from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch.nn as nn
import torch.optim as optim
from modules.datamanager.raster.raster_loader_1dcnn import CNNRasterDataManager
from modules.utils.run_util import check_device
import logging
import torch
from modules.lib.constants import PICNN1D_V1
import time
import json
import numpy as np
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import format_flops
from torch.utils.flop_counter import FlopCounterMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PI1DCNN_ModelWrapper")
model_name  = PICNN1D_V1

class PICNN1DModel(nn.Module):
    def __init__(self, steps, features, outputs):
        super(PICNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * steps, 32)
        self.bn_fc1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, outputs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        # First convolutional block with batch norm
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second convolutional block with batch norm
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        
        # First fully connected block with batch norm and dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Second fully connected block with batch norm and dropout
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Third fully connected block with batch norm
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = self.relu(x)
        
        # Output layer
        x = self.fc4(x)
        return x
            
class PICNN1DModelWrapper(ModelWrapper):
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        self.device = check_device()
        self.lag = config.lag
        self.steps = 1
        self.features = self.lag * 3
        self.outputs = 581067
        self.validation_event = self.config.fold  + 1
        self.physics_weight = self.config.args.get('physics_weight', 0.5)
        self.tuning_mode = config.args.get('tuning_mode', True)
    
    def create_dataset(self):
        logger.info("Creating dataset")
        self.data_manager = CNNRasterDataManager(self.config.lag, self.config.batch_size, pinn=True, validation_event=self.validation_event, tuning_mode=self.tuning_mode)
        self.features = self.data_manager.features
        self.outputs = self.data_manager.outputs
        logger.info(f"Features: {self.features}, Outputs: {self.outputs}")
        
    def init_model(self) -> bool:
        try:
            self.create_dataset()
            self.model = PICNN1DModel(self.steps, self.features, self.outputs).to(self.device)
            self.loss_fn = nn.MSELoss()
            self.physics_loss_fn = self.loss_fn_def
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            return True
        
        except Exception as e:
            logger.error(f"Error creating 1DCNN model: {e}")
            return False
        
    def train(self, run_dir: str, tuning_mode:True) -> str:
        return super().train(run_dir, tuning_mode)
        
    def train_model(self, run_dir: str, tuning_mode = True):
        device = self.device
        history = {
            "loss": [],
            "val_loss": [],
            "train_time": None
        }
        start_time = time.time()
        best_val_loss = float("inf")
        epochs_no_improvement = 0
        best_epoch = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            valid_batches = 0
            self.model.train()
            
            if epoch > 0:
                self.data_manager.shuffle_training_data(epoch) 
                
            for idx, t_indices in enumerate(self.data_manager.train_idx):
                self.optimizer.zero_grad()
                if idx == 0 and epoch == 0:
                    logger.info("Starting memory profiling for the first batch")
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                                profile_memory=True, 
                                on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir)) as prof:
                        xt, yt, yt_minus1, yt_plus1, bct, bct_plus1  = self.data_manager.get_batch(t_indices)
                        pred = self.model(xt)
                        batch_loss = self.physics_loss_fn(pred, yt, yt_minus1, yt_plus1, bct, bct_plus1)
                        batch_loss.backward()
                        self.optimizer.step()
                        prof.step()
                    self.profiler= prof
                else:
                    xt, yt, yt_minus1, yt_plus1, bct, bct_plus1  = self.data_manager.get_batch(t_indices)
                    pred = self.model(xt)
                    batch_loss = self.physics_loss_fn(pred, yt, yt_minus1, yt_plus1, bct, bct_plus1)
                    batch_loss.backward()
                    self.optimizer.step()
                    
                del pred
                del yt_plus1
                del yt_minus1
                del bct
                del bct_plus1
                del yt
                del xt
                epoch_loss += batch_loss.item()
                valid_batches += 1  # Increment valid batch counter
                torch.cuda.empty_cache()
                logger.info(f"Batch train loss: {batch_loss.item()}")
                
            # Divide by actual number of valid batches processed
            epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
            history["loss"].append(epoch_loss)
            
            
        
            # Epoch Validation
            if tuning_mode:
                val_loss = 0
                valid_val_batches = 0
                self.model.eval()
                for idx, t_indices in enumerate(self.data_manager.validation_idx):
                    xt, yt, yt_plus1, bct, bct_plus1  = self.data_manager.get_batch(t_indices)
                    if xt is None or yt is None:
                        logger.warning(f"Error getting validation batch {idx}, skipping")
                        continue

                    with torch.no_grad():
                        pred_val = self.model(xt)
                        # If pred < 0.2 then set to 0
                        pred_val = torch.where(pred_val < 0.2, torch.tensor(0.0).to(device), pred_val)
                        batch_val_loss = self.loss_fn(pred_val, yt).item()
                        del pred_val
                        del yt
                        del yt_plus1
                        del bct
                        del bct_plus1
                        del xt
                        
                        val_loss += batch_val_loss
                        valid_val_batches += 1  # Increment valid validation batch counter
                        logger.info(f"Batch validation loss: {batch_val_loss} ")
                        torch.cuda.empty_cache()  # Add this at strategic points
                        
                # Divide by actual number of valid validation batches processed
                epoch_val_loss = val_loss / valid_val_batches if valid_val_batches > 0 else float('inf')
                history["val_loss"].append(epoch_val_loss)
                logger.info(f"Epoch {epoch} loss: {epoch_loss}  validation loss: {epoch_val_loss} ")
                
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improvement = 0
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        logger.info(f"Best validation loss: {best_val_loss} at epoch {best_epoch}")
                        break
            else:
                logger.info(f"Epoch {epoch} loss: {epoch_loss}")
    
        torch.cuda.empty_cache()   
        end_time = time.time()
        train_time = end_time - start_time
        logger.info(f"Training time: {train_time}")
        history["train_time"] = train_time
        
        # Add required metrics to history
        if tuning_mode:
            best_val_rmse = np.sqrt(best_val_loss) if best_val_loss != float('inf') else None
            history["best_val_rmse"] = best_val_rmse
            history["best_epoch"] = best_epoch
            
        # Create hyperparameters dictionary
        hyperparameters = {
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "lag": self.config.lag,
            "horizon": self.config.horizon,
            "physics_weight": self.physics_weight
        }
        history["hyperparameters"] = json.dumps(hyperparameters)
        
        model_file = None
        if not tuning_mode:
            logger.info("Loading best model state")
            model_state = self.model.state_dict().copy()
            logger.info("Saving model with best validation loss")
            model_file = self.save_model_checkpoint(self.config.run_id, run_dir, model_state, self.config)
        return history, train_time, model_file
        
    def test_model(self):
        return super().test_model()
        
    def loss_fn_def(self, y_hat_t, y_t, yt_minus1, yt_plus1=None, bct=None, bct_plus1=None):
        mse_loss = self.loss_fn(y_hat_t, y_t)
        if yt_plus1 is None or bct is None or bct_plus1 is None:
            return mse_loss
        
        delta_x = 5
        delta_y = 5
        n_x = 911
        n_y = 651
        delta_t_val = 15 * 60  # 15 minutes in seconds
        area = delta_x * delta_y * n_x * n_y
        
        # VECTORIZED: Calculate volumes across entire batch at once
        vt_minus1 = delta_x * delta_y * torch.sum(yt_minus1, dim=1)  # [batch_size]
        vt_plus1 = delta_x * delta_y * torch.sum(yt_plus1, dim=1)  # [batch_size]
        v_hat_t = delta_x * delta_y * torch.sum(y_hat_t, dim=1)  # [batch_size]
        
        # VECTORIZED: Sum boundary conditions for each sample
        bc_t_sum = torch.sum(bct, dim=1)  # [batch_size]
        bc_t_plus_1_sum = torch.sum(bct_plus1, dim=1)  # [batch_size]
        
        # VECTORIZED: Physics calculations on entire batch at once
        relu_arg1 = v_hat_t - vt_minus1 - delta_t_val * bc_t_sum
        term2 = ((torch.relu(relu_arg1)) / area)**2
        
        relu_arg2 = vt_plus1 - v_hat_t - delta_t_val * bc_t_plus_1_sum
        term3 = ((torch.relu(relu_arg2)) / area)**2
        
        # Mean across batch
        physics_loss = torch.mean(term2 + term3)
        
        # Final loss
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        return total_loss
    
    def calculate_flops(self):
        try:
            t_indices = self.data_manager.train_idx[0]
            input_batch  = self.data_manager.get_batch(t_indices)[0]

            # Create a sample input for the model
            input_batch = torch.tensor(input_batch).to(self.device)
            sample_input = input_batch[0].unsqueeze(0)
        
            # Use FlopCounterMode to count FLOPS
            with FlopCounterMode(self.model) as counter:
                _ = self.model(sample_input)
                
            flops = counter.get_total_flops()
            logger.info(f"FLOPS: {flops}")
            flops_str = format_flops(flops)
            logger.info(f"Model FLOPS: {flops_str}")
            
            return flops
        except Exception as e:
            logger.error(f"Error calculating FLOPS: {e}")
            return None