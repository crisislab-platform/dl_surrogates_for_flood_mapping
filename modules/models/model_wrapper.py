from dataclasses import dataclass
from logging import config
import os
import logging
import time
import torch
import numpy as np
import time
from torch.utils.flop_counter import FlopCounterMode
from torch.profiler import profile, ProfilerActivity
from tqdm.asyncio import tqdm
from modules.lib.constants import MODEL_CHECKPOINT_DIR
from modules.utils.model_util import profiler_analysis, format_flops, save_prediction_map
from modules.datamanager.datamanager import DataManager
import json
import pandas as pd
from modules.lib.constants import RUN_DIR
import rasterio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model")

# Unified Model configuration
@dataclass
class Config:
    model_name: str
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    dropout: float = 0.2
    epochs: int = 10
    patience: int = 2
    run_id: str = None
    run_dir: str = None
    save_model: bool = False
    save_predictions: bool = False
    train_events: list = None
    validation_events: list = None
    test_events: list = None
    indices_per_timestep: int = 1
    tuning_mode: bool = True
    study_area: str = "carlisle"
    args: dict = None
    

class ModelWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.model_name = config.model_name
        self.train_events = config.train_events
        self.validation_events = config.validation_events
        self.test_events = config.test_events
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.scheduler = None
        self.dropout = config.args.get('dropout', 0.2)
        self.save_model = config.args.get('save_model', False)
        self.is_save_predictions = config.args.get('save_predictions', False)
        self.loss_fn = None
        self.data_manager:DataManager = None
        self.optimizer = None
        self.device = None
        self.profiler = None
        self.do_profile = False
        self.tuning_mode = config.tuning_mode
        self.study_area = config.study_area
        
    def setup_file_logging(self, log_file_path: str):
        """Mirror training logs to a per-run file."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == os.path.abspath(log_file_path):
                root_logger.removeHandler(handler)
                handler.close()

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
        logger.info(f"Training log file initialized at {log_file_path}")
        
        
    def init_model(self) -> bool:
        pass
    
    def create_dataset(self):
        pass
    
    def train_model(self, run_dir: str, tuning_mode = True):
        self.setup_file_logging(os.path.join(run_dir, "training.log"))
        torch.cuda.empty_cache()
        logger.info(f"Saving model training history {self.config.save_model}")
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
            
            tqdm_loader = tqdm(self.data_manager.train_data_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for input_batch, output_batch in tqdm_loader:
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                
                #check if a batch has five dimentions. If so make it four
                if len(input_batch.shape) == 5:
                    B, T, C, H, W = input_batch.shape
                    input_batch = input_batch.view(B * T, C, H, W)
                if len(output_batch.shape) == 5:
                    B, T, C, H, W = output_batch.shape
                    output_batch = output_batch.view(B * T, C, H, W)
                
                
                if input_batch is None or output_batch is None:
                    logger.warning(f"Error getting batch skipping")
                    continue
            
                pred = self.model(input_batch)
                if pred.shape != output_batch.shape:
                    output_batch = output_batch.squeeze(1)
                    B, H, W = output_batch.shape
                    pred = pred.view(B, H, W)
                batch_loss = self.loss_fn(pred, output_batch)
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() # Clear gradients for the next iteration
                tqdm_loader.set_postfix({"Batch Loss": batch_loss.item()})
                epoch_loss += batch_loss.item()
                valid_batches += 1  # Increment valid batch counter
                if self.scheduler:
                    self.scheduler.step()
                    
                if valid_batches % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs} batch {valid_batches}: batch_loss={batch_loss.item():.6f}"
                )
                
            # Divide by actual number of valid batches processed
            epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
            history["loss"].append(epoch_loss)

            # Epoch Validation
            if tuning_mode:
                val_loss = 0
                valid_val_batches = 0
                self.model.eval()
                val_tqdm_loader = tqdm(self.data_manager.validation_data_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} - Validation")
                for input_batch, output_batch in val_tqdm_loader:
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    with torch.no_grad():
                        pred_val = self.model(input_batch)
                        if pred_val.shape != output_batch.shape:
                            output_batch = output_batch.squeeze(1)
                            B, H, W = output_batch.shape
                            pred_val = pred_val.view(B, H, W)
                        batch_val_loss = self.loss_fn(pred_val, output_batch).item()
                        val_loss += batch_val_loss
                        valid_val_batches += 1
                        val_tqdm_loader.set_postfix({"Batch Val Loss": batch_val_loss})
                        
                        if  valid_val_batches % 10 == 0:
                            logger.info(
                            f"Epoch {epoch + 1}/{self.config.epochs} batch {valid_val_batches}: batch_loss={batch_val_loss:.6f}"
                        )
                        
                # Divide by actual number of valid validation batches processed
                epoch_val_loss = val_loss / valid_val_batches if valid_val_batches > 0 else float('inf')
                history["val_loss"].append(epoch_val_loss)
                logger.info(f"Epoch {epoch + 1} loss: {epoch_loss}  validation loss: {epoch_val_loss} ")
                            
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improvement = 0
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        logger.info(f"Best validation loss: {best_val_loss} at epoch {best_epoch}")
                        break

                torch.cuda.empty_cache() 
                if self.data_manager.indices_per_timestep > 1: 
                    self.validate_reconstruction(self.validation_idx)
            else:
                logger.info(f"Epoch {epoch + 1} loss: {epoch_loss} ")
            
        # Add required metrics to history
        if tuning_mode:
            best_val_rmse = np.sqrt(best_val_loss) if best_val_loss != float('inf') else None
            history["best_val_rmse"] = best_val_rmse
            history["best_epoch"] = best_epoch
            
        # Create hyperparameters dictionary
        hyperparameters = self.create_hyperparameters_dict()
        history["hyperparameters"] = json.dumps(hyperparameters)
                
        end_time = time.time()
        train_time = end_time - start_time
        logger.info(f"Training time: {train_time}")
        history["train_time"] = train_time
    
        # Save the model state
        model_file = None
        if self.config.save_model:
            logger.info(f"Saving model state to {run_dir}")
            model_state = self.model.state_dict().copy()
            model_file = self.save_model_checkpoint(model_state, self.config)
        return history, train_time, model_file

    
    def train(self, run_dir: str, tuning_mode = True):
        if tuning_mode:
            logger.info("Tuning mode is enabled, skipping profiling")
            return self.train_model(run_dir, tuning_mode)

        logger.info("Starting model training with memory profiling for first batch")
        history, train_time, model_file = self.train_model(run_dir, tuning_mode)
        
        if self.profiler is not None:
            logger.info("Training completed, analyzing memory usage")
            key_averages = self.profiler.key_averages()
            analysis_results = profiler_analysis(key_averages)
            logger.info(f"Memory profiling results: {key_averages.table(sort_by='cuda_memory_usage', row_limit=10)}")
            logger.info(f"Profiler analysis results: {analysis_results}")
            history['memory'] = analysis_results
        else:
            history['memory'] = "No profiling data available"
            
        return history, train_time, model_file

    
    def inference(self, input_batch):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_batch)
        return pred
    
    def validate_reconstruction(self, validation_idx):
        #implement
        pass
    
    def test_model(self):
        logger.info("Testing model")
        self.model.eval()
           
        start_time = time.time()
        total_tp = 0
        total_fn = 0
        total_fp = 0
        total_tn = 0
        total_mse = 0
        total_rmse = 0
        total_mrmse = 0
        prof = None
        tqdm_loader = tqdm(self.data_manager.test_data_loader, desc=f"Testing on test data")
        
        for input_batch, output_batch in tqdm_loader:
            with torch.no_grad():
                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)
                if self.do_profile and prof == None:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
                        batch_pred = self.inference(input_batch)
                else:
                    batch_pred = self.inference(input_batch)

                # Handle single pixel output - no squeezing needed for 1D output
                if len(batch_pred.shape) > 1 and batch_pred.shape[1] > 1:
                    batch_pred = batch_pred.squeeze(1)
                if len(output_batch.shape) > 1 and output_batch.shape[1] > 1:
                    output_batch = output_batch.squeeze(1)

                # If pred < 0 then set to 0
                model_pred = torch.where(batch_pred < 0, torch.tensor(0.0).to(self.device), batch_pred)
                ref_out = torch.where(output_batch < 0, torch.tensor(0.0).to(self.device), output_batch)
                
                loss = self.loss_fn(model_pred, ref_out)
                mse = loss.item()
                rmse = np.sqrt(mse)
                total_mse += mse
                total_rmse += rmse

                # Calculate mRMSE for wet cells
                mRMSE = self.mRMSE_fn(model_pred, ref_out)
                total_mrmse += mRMSE

                threshold = 0.3
                pred_binary = (model_pred > threshold).float()
                ref_binary = (ref_out > threshold).float()
                
                batch_tp = ((pred_binary == 1) & (ref_binary == 1)).sum().item()
                batch_tn = ((pred_binary == 0) & (ref_binary == 0)).sum().item()
                batch_fp = ((pred_binary == 1) & (ref_binary == 0)).sum().item()
                batch_fn = ((pred_binary == 0) & (ref_binary == 1)).sum().item()
                total_tp += batch_tp
                total_tn += batch_tn
                total_fp += batch_fp
                total_fn += batch_fn
                
                tqdm_loader.set_postfix({"Batch MSE": mse, "Batch RMSE": rmse, "Batch mRMSE": mRMSE, "Batch TP": batch_tp, "Batch TN": batch_tn, "Batch FP": batch_fp, "Batch FN": batch_fn})

        end_time = time.time()
        mse = total_mse / len(self.data_manager.test_idx) if len(self.data_manager.test_idx) > 0 else float('inf')
        rmse = total_rmse / len(self.data_manager.test_idx) if len(self.data_manager.test_idx) > 0 else float('inf')
        mRMSE = total_mrmse / len(self.data_manager.test_idx) if len(self.data_manager.test_idx) > 0 else float('inf')
        
        logger.info(f"Confusion Matrix at {threshold}m threshold - TP: {total_tp}, TN: {total_tn}, FP: {total_fp}, FN: {total_fn}")
        
        # Hit Rate
        hit_rate = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        logger.info(f"Hit Rate: {hit_rate}")
        
        # Critical Success Index (CSI)
        csi = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        logger.info(f"Critical Success Index (CSI): {csi}")
        
        # F2 Score
        f2_score = (total_tp - total_fn) / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        # F3 Score
        f3_score = (total_tp - total_fp) / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
                
        pred_time = end_time - start_time
        logger.info(f"Validation prediction_time:{pred_time} loss MSE: {mse} RMSE: {rmse}  mRMSE: {mRMSE}")
        
        flops = self.calculate_flops()
        analysis_results = profiler_analysis(prof.key_averages())
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mRMSE": mRMSE,
            "hit_rate": hit_rate,
            "csi": csi,
            "f2_score": f2_score,
            "f3_score": f3_score,
            "pred_time": pred_time,
            "flops": flops,
            "pred_memory_usage": analysis_results
        }
        
        logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        return metrics
     
   
    def mRMSE_fn(self, pred, ref_out):
        # Define threshold for wet cells (typically > 0.01m is considered wet)
        threshold = 0.3
        
        # Create binary masks
        pred_wet = (pred > threshold).float()
        ref_wet = (ref_out > threshold).float()
        
        # Calculate RMSE for wet cells only
        pred_wet_values = pred * ref_wet
        ref_wet_values = ref_out * ref_wet
        wet_loss = self.loss_fn(pred_wet_values, ref_wet_values)
        rmse_wet = np.sqrt(wet_loss.item())
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

    def save_model_checkpoint(self):
        try:
            model_file = os.path.join(MODEL_CHECKPOINT_DIR, self.model_name, f"model.pth")
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.epochs,
                'run_id': self.config.run_id,
            }, model_file) 
            
            return model_file      
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None
        
    def save_predictions(self, pred, idx):
        output_dir = os.path.join(RUN_DIR, "output_maps", self.config.model_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving prediction map to {output_dir}")
        save_prediction_map(pred, output_dir, idx, self.config.model_name)
        
        
    def create_hyperparameters_dict(self):
        hyperparameters = {
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "lag": self.config.lag,
            "horizon": self.config.horizon
        }
        return hyperparameters