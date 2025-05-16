from dataclasses import dataclass
import os
import logging
import time
import torch
from torch.profiler import profile, ProfilerActivity
import numpy as np
import time
from torch.utils.flop_counter import FlopCounterMode
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map
from modules.datamanager.datamanager import DataManager
import json

logger = logging.getLogger("Model")

# Unified Model configuration
@dataclass
class ModelConfig:
    """Unified configuration class for model parameters and training"""
    model_name: str
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    epochs: int = 10
    patience: int = 2
    run_id: str = None
    run_dir: str = None
    args: dict = None
    fold: int = None #validation fold

class ModelWrapper:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.train_dataset = None
        self.val_dataset = None
        self.x_test = None
        self.y_test = None
        self.grid_height = None
        self.grid_width = None
        self.train_steps = None
        self.val_steps = None
        self.model = None
        self.loss_fn = None
        self.data_manager:DataManager = None
        self.optimizer = None
        self.device = None
        
    def init_model(self) -> bool:
        pass
    
    def create_dataset(self):
        pass
    
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
        if not tuning_mode:
            best_model_state = None
            best_optimizer_state = None
       
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            valid_batches = 0
            self.model.train()
            for idx, t_indices in enumerate(self.data_manager.train_idx):
                self.optimizer.zero_grad()
                input_batch, output_batch = self.data_manager.get_batch(t_indices)
                if input_batch is None or output_batch is None:
                    logger.warning(f"Error getting batch {idx}, skipping")
                    continue
                input_in_batch = input_batch.to(device)
                output_in_batch = output_batch.to(device)
                pred = self.model(input_in_batch.float())
                output_in_batch = output_in_batch.float()
                batch_loss = self.loss_fn(pred, output_in_batch)
                epoch_loss += batch_loss.item()
                valid_batches += 1  # Increment valid batch counter
                batch_loss.backward()
                self.optimizer.step()
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
                    input_batch, output_batch = self.data_manager.get_batch(t_indices)
                    if input_batch is None or output_batch is None:
                        logger.warning(f"Error getting validation batch {idx}, skipping")
                        continue
                    input_in_batch = input_batch.to(device)
                    output_in_batch = output_batch.to(device)

                    with torch.no_grad():
                        pred_val = self.model(input_in_batch)
                        # If pred < 0.2 then set to 0
                        pred_val = torch.where(pred_val < 0.2, torch.tensor(0.0).to(device), pred_val)
                        batch_val_loss = self.loss_fn(pred_val, output_in_batch).item()
                        val_loss += batch_val_loss
                        valid_val_batches += 1  # Increment valid validation batch counter
                        logger.info(f"Batch validation loss: {batch_val_loss} ")
                        
                # Divide by actual number of valid validation batches processed
                epoch_val_loss = val_loss / valid_val_batches if valid_val_batches > 0 else float('inf')
                history["val_loss"].append(epoch_val_loss)
                logger.info(f"Epoch {epoch} loss: {epoch_loss}  validation loss: {epoch_val_loss} ")
            
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    if not tuning_mode:
                        best_model_state = self.model.state_dict().copy()
                        best_optimizer_state = self.optimizer.state_dict().copy()
                    epochs_no_improvement = 0
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        logger.info(f"Best validation loss: {best_val_loss} at epoch {best_epoch}")
                        break
            torch.cuda.empty_cache()   
            
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
                
                
        end_time = time.time()
        train_time = end_time - start_time
        logger.info(f"Training time: {train_time}")
        history["train_time"] = train_time
    
        # Save the model state
        model_file = None
        if not tuning_mode and best_model_state is not None:
                logger.info("Loading best model state")
                self.model.load_state_dict(best_model_state)
                logger.info("Saving model with best validation loss")
                model_file = self.save_model_checkpoint(self.config.run_id, run_dir, best_model_state, best_optimizer_state, self.config)
        return history, train_time, model_file
    
    def train(self, run_dir: str, tuning_mode = True):
        if tuning_mode:
            return self.train_model(run_dir, tuning_mode)
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir)) as prof:
            history, train_time, model_file = self.train_model(run_dir, tuning_mode)
        key_averages = prof.key_averages()
        analysis_results = profiler_analysis(key_averages)
        logger.info(f"Memory profiling results: {key_averages.table(sort_by='cuda_memory_usage', row_limit=10)}")
        logger.info(f"Profiler analysis results: {analysis_results}")
        history['memory'] = analysis_results
        return history, train_time, model_file
    
    def test_model(self):
        logger.info("Testing model")
        self.model.eval()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
            with torch.no_grad():
                input_data = (self.data_manager.test_input).to(self.device)
                output_data = (self.data_manager.test_output).to(self.device)
                start_time = time.time()
                pred = self.model(input_data)
                # If pred < 0.2 then set to 0
                pred = torch.where(pred < 0.2, torch.tensor(0.0).to(self.device), pred)
                end_time = time.time()
                ref_out = output_data.float()
                loss = self.loss_fn(pred, ref_out)
                mse = loss.item()
                rmse = np.sqrt(mse)
                
                # Calculate mRMSE for wet cells
                mRMSE = self.mRMSE_fn(pred, ref_out)

                # Calculate NSE
                observed = ref_out
                predicted = pred
                nse = self.nse_fn(observed, predicted)
                
                pred_time = end_time - start_time
                logger.info(f"Validation prediction_time:{pred_time} loss MSE: {mse} RMSE: {rmse}  NSE: {nse} mRMSE: {mRMSE}")
                flops = self.calculate_flops()
                self.save_predictions(pred)
                
       
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
    
    def nse_fn(self, observed, predicted):
        observed_mean = torch.mean(observed)
        numerator = torch.sum((observed - predicted) ** 2)
        denominator = torch.sum((observed - observed_mean) ** 2)
        nse = 1 - (numerator / denominator)
        nse = nse.item()
        return nse 
    
    def mRMSE_fn(self, pred, ref_out):
        # Define threshold for wet cells (typically > 0.01m is considered wet)
        threshold = 0.02
        
        # Create binary masks
        pred_wet = (pred > threshold).float()
        ref_wet = (ref_out > threshold).float()
        
        # Calculate RMSE for wet cells only
        pred_wet_values = pred * ref_wet
        ref_wet_values = ref_out * ref_wet
        wet_loss = self.loss_fn(pred_wet_values, ref_wet_values)
        rmse_wet = np.sqrt(wet_loss.item())
        logger.info(f"Wet cells RMSE: {rmse_wet}")
        return rmse_wet
    
    def calculate_flops(self):
        try:
            t_indices = self.data_manager.train_idx[0]
            input_batch, _  = self.data_manager.get_batch(t_indices)

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

    def save_model_checkpoint(self, run_id, run_dir, best_model_state, best_optimizer_state, config):
        try:
            model_file = os.path.join(run_dir, f"{config.model_name}_{run_id}.pth")
            torch.save({
                'model_state_dict': best_model_state,
                'optimizer_state_dict': best_optimizer_state,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.epochs,
                'run_id': run_id,
            }, os.path.join(run_dir, model_file)) 
            return model_file      
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None
        
    def save_predictions(self, pred):
        idx = 146
        pred_max = pred.detach().cpu().numpy()[idx]
        output_dir = os.path.join(self.config.run_dir, "output_maps")
        save_prediction_map(pred_max, output_dir, idx)

