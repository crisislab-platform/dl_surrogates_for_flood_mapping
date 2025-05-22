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
            self.loss_fn = self.loss_fn_def
            self.val_loss_fn = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
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
        best_model_state = None
        best_optimizer_state = None
        best_epoch = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            valid_batches = 0
            self.model.train()
            for idx, t_indices in enumerate(self.data_manager.train_idx):
                self.optimizer.zero_grad()
                xt, yt, yt_plus1, bct, bct_plus1  = self.data_manager.get_batch(t_indices)
                
                if xt is None or yt is None:
                    logger.warning(f"Error getting batch {idx}, skipping")
                    continue
                
                input_in_batch = xt.to(device)
                output_in_batch = yt.to(device)
                del xt 
                del yt
                
                bct_tensor = bct.to(device)
                bct_plus1_tensor = bct_plus1.to(device)
                yt_plus1_tensor = yt_plus1.to(device)
                
                del yt_plus1
                del bct
                del bct_plus1
                
                pred = self.model(input_in_batch.float())
                output_in_batch = output_in_batch.float()
                batch_loss = self.loss_fn(pred, output_in_batch, yt_plus1_tensor, bct_tensor, bct_plus1_tensor)
                
                del pred
                del output_in_batch
                del bct_tensor
                del bct_plus1_tensor
                del yt_plus1_tensor
                del input_in_batch
                
                epoch_loss += batch_loss.item()
                valid_batches += 1  # Increment valid batch counter
                batch_loss.backward()
                self.optimizer.step()
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
                    input_in_batch = xt.to(device)
                    output_in_batch = yt.to(device)
                    
                    del xt
                    del yt
                    
                    bct_tensor = bct.to(device)
                    bct_plus1_tensor = bct_plus1.to(device)
                    yt_plus1_tensor = yt_plus1.to(device)
                    
                    del yt_plus1
                    del bct
                    del bct_plus1

                    with torch.no_grad():
                        pred_val = self.model(input_in_batch)
                        # If pred < 0.2 then set to 0
                        pred_val = torch.where(pred_val < 0.2, torch.tensor(0.0).to(device), pred_val)
                        batch_val_loss = self.val_loss_fn(pred_val, output_in_batch).item()
                        del pred_val
                        del output_in_batch
                        del bct_tensor
                        del bct_plus1_tensor
                        del yt_plus1_tensor
                        
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
                    if not tuning_mode:
                        best_model_state = self.model.state_dict().copy()
                        best_optimizer_state = self.optimizer.state_dict().copy()
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        logger.info(f"Best validation loss: {best_val_loss} at epoch {best_epoch}")
                        break
    
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
            "horizon": self.config.horizon
        }
        history["hyperparameters"] = json.dumps(hyperparameters)
        
        model_file = None
        if not tuning_mode:
            if best_model_state is not None:
                logger.info("Loading best model state")
                self.model.load_state_dict(best_model_state)
                logger.info("Saving model with best validation loss")
                model_file = self.save_model_checkpoint(self.config.run_id, run_dir, best_model_state, best_optimizer_state, self.config)
        return history, train_time, model_file
        
    def test_model(self):
        return super().test_model()
        
    def loss_fn_def(self, y_hat_t, y_t, y_t_plus_1=None, bc_t=None, bc_t_plus_1=None):
        # Standard MSE loss - this already handles batch properly
        mse_loss = nn.MSELoss()(y_hat_t, y_t)
        
        if y_t_plus_1 is None or bc_t is None or bc_t_plus_1 is None:
            return mse_loss

        # Physics weight to control contribution
        physics_weight = 1  # Small weight to start with
        
        delta_x = 5
        delta_y = 5
        n_x = 911
        n_y = 651
        delta_t_val = 15 * 60  # 15 minutes in seconds
        area = delta_x * delta_y * n_x * n_y
        
        # VECTORIZED: Calculate volumes across entire batch at once
        v_t = delta_x * delta_y * torch.sum(y_t, dim=1)  # [batch_size]
        v_t_plus_1 = delta_x * delta_y * torch.sum(y_t_plus_1, dim=1)  # [batch_size]
        v_hat_t = delta_x * delta_y * torch.sum(y_hat_t, dim=1)  # [batch_size]
        
        # VECTORIZED: Sum boundary conditions for each sample
        bc_t_sum = torch.sum(bc_t, dim=1)  # [batch_size]
        bc_t_plus_1_sum = torch.sum(bc_t_plus_1, dim=1)  # [batch_size]
        
        # VECTORIZED: Physics calculations on entire batch at once
        relu_arg1 = v_hat_t - v_t - delta_t_val * bc_t_sum
        term2 = ((torch.relu(relu_arg1)) / area)**2
        
        relu_arg2 = v_t_plus_1 - v_hat_t - delta_t_val * bc_t_plus_1_sum
        term3 = ((torch.relu(relu_arg2)) / area)**2
        
        # Mean across batch
        physics_loss = torch.mean(term2 + term3)
        
        # Final loss
        total_loss = mse_loss + physics_weight * physics_loss
        
        return total_loss