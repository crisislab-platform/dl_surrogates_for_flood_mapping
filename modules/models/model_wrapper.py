from dataclasses import dataclass
from logging import config
import os
import logging
import time
from decorator import contextmanager
import torch
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    do_profile: bool = False
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
    random_seed: int = 42
    patch_domain: bool = False
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
        self.do_profile = config.do_profile
        self.tuning_mode = config.tuning_mode
        self.study_area = config.study_area
        self.random_seed = config.random_seed
        self.autoregressive_model = False
        
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
    
    def train_model(self, run_dir: str, tuning_mode = False):
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
        self.best_val_loss = float("inf")
        self.epochs_no_improvement = 0
        self.best_epoch = 0
        epochs_lr_reduced = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            valid_batches = 0
            self.model.train()
            
            tqdm_loader = tqdm(self.data_manager.train_data_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for input_batch, output_batch in tqdm_loader:
                input_batch = input_batch.to(device, non_blocking=True)
                output_batch = output_batch.to(device, non_blocking=True)

                #check if a batch has five dimentions.
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
                self.optimizer.zero_grad() # Clear gradients for the next iteration
                batch_loss.backward()
                self.optimizer.step()

                tqdm_loader.set_postfix({"Batch Loss": batch_loss.item()})
                epoch_loss += batch_loss.item()
                valid_batches += 1  # Increment valid batch counter
                if valid_batches % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs} batch {valid_batches}: batch_loss={batch_loss.item():.6f}"
                )
                
            # Divide by actual number of valid batches processed
            epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
            history["loss"].append(epoch_loss)
            
            if self.config.tuning_mode:
                validation_loss = self.validation(epoch)
                history["val_loss"].append(validation_loss)
                logger.info(f"Epoch {epoch + 1} loss: {epoch_loss}  validation loss: {validation_loss} ")

                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        lr_before = self.optimizer.param_groups[0]['lr']
                        self.scheduler.step(validation_loss)
                        lr_after = self.optimizer.param_groups[0]['lr']
                        
                        if lr_after < lr_before:
                            logger.info(f"LR reduced: {lr_before:.2e} → {lr_after:.2e}")
                            epochs_lr_reduced.append([epoch, lr_after])
                        else:
                            logger.info(f"LR unchanged: {lr_after:.2e} (validation loss: {validation_loss:.4f})")
                    else:
                        self.scheduler.step()
                            
                if validation_loss < self.best_val_loss:
                    self.best_val_loss = validation_loss
                    self.epochs_no_improvement = 0
                    self.best_epoch = epoch
                else:
                    self.epochs_no_improvement += 1
                    if self.epochs_no_improvement >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        logger.info(f"Best validation loss: {self.best_val_loss} at epoch {self.best_epoch}")
                        break
            else:
                logger.info(f"Epoch {epoch + 1} loss: {epoch_loss} ")
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        lr_before = self.optimizer.param_groups[0]['lr']
                        self.scheduler.step(epoch)
                        lr_after = self.optimizer.param_groups[0]['lr']
                        
                        if lr_after < lr_before:
                            logger.info(f"LR reduced: {lr_before:.2e} → {lr_after:.2e}")
                            
                        else:
                            logger.info(f"LR unchanged: {lr_after:.2e} (epoch loss: {epoch_loss:.4f})")
                            
                    else:
                        self.scheduler.step()
        
            
        # Add required metrics to history
        if tuning_mode:
            history["best_val_loss"] = self.best_val_loss
            history["best_epoch"] = self.best_epoch
            history["epochs_lr_reduced"] = epochs_lr_reduced
            
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
    
    def train(self, run_dir: str, tuning_mode = False):
        if tuning_mode:
            logger.info("Tuning mode is enabled, skipping profiling")
            return self.train_model(run_dir, tuning_mode)

        logger.info("Starting model training with memory profiling for first batch")
        history, train_time, model_file = self.train_model(run_dir, tuning_mode)
            
        return history, train_time, model_file

    
    def inference(self, input_batch):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_batch)
        return pred
    
    @contextmanager
    def track_peak_gpu_memory(self): 
        torch.cuda.reset_peak_memory_stats()
        class Result:
            peak_mb = None
        
        yield Result
        Result.peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        
    def validation(self, epoch):
        torch.cuda.empty_cache() 
        if self.data_manager.patch_domain or self.autoregressive_model: 
            return self.sequential_validation(self.data_manager.validation_idx, testing_mode=False)
        
        #Epoch Validation
        self.model.eval()
        val_loss = 0
        valid_val_batches = 0
        val_tqdm_loader = tqdm(self.data_manager.validation_data_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} - Validation")
        for input_batch, output_batch in val_tqdm_loader:
            input_batch = input_batch.to(self.device)
            output_batch = output_batch.to(self.device)
            if len(input_batch.shape) == 5:
                B, T, C, H, W = input_batch.shape
                input_batch = input_batch.view(B * T, C, H, W)
            if len(output_batch.shape) == 5:
                B, T, C, H, W = output_batch.shape
                output_batch = output_batch.view(B * T, C, H, W)
            with torch.no_grad():
                pred_val = self.model(input_batch)
                if pred_val.shape != output_batch.shape:
                    output_batch = output_batch.squeeze(1)
                    B, H, W = output_batch.shape
                    pred_val = pred_val.view(B, H, W)
                batch_val_loss = self.loss_fn(pred_val, output_batch).item()
                val_loss += batch_val_loss
                valid_val_batches += 1
                
            if valid_val_batches % 10 == 0: 
                logger.info(f"Epoch {epoch + 1} batch {valid_val_batches}: batch_val_loss={batch_val_loss:.6f}")    
        # Divide by actual number of valid validation batches processed
        epoch_val_loss = val_loss / valid_val_batches if valid_val_batches > 0 else float('inf')
        return epoch_val_loss
        
    @torch.inference_mode()
    def sequential_validation(self, batch_indices, testing_mode=False):
        logger.info("Testing model")
        event_metrics = {}
        
        running_batch_loss = 0
        running_mse = 0
        num_batches = 0
    
        #Organize batch indices by event and timestep
        event_timestep_dict = {}
        batch_indices.sort()
        
        for idx in batch_indices:
            event_id = self.data_manager.find_event_id(idx)
            timestep, tile_group = self.data_manager.find_local_indices(event_id, idx)
            if event_id not in event_timestep_dict:
                event_timestep_dict[event_id] = {}
            
            if timestep not in event_timestep_dict[event_id]:
                event_timestep_dict[event_id][timestep] = []
                
            event_timestep_dict[event_id][timestep].append(idx)
        
        
        #put the model in evaluation mode
        self.model.eval()
        metrics = ["mse", "rmse", "mRMSE", "hit_rate", "csi", "f2_score", "timestep_time"]
        with torch.no_grad():
            for event_id, timestep_dict in event_timestep_dict.items():
                start_time_event = time.time() #start time for this event
                
                for metric in metrics:
                    event_metrics.setdefault(metric, {})
                    event_metrics[metric][event_id] = []
                event_metrics.setdefault("event_time", {})
                event_metrics.setdefault("peak_volume_timestep", {})
                event_metrics.setdefault("true_peak_volume_timestep", {})
                event_metrics.setdefault("peak_volume_timing_error", {})
                true_peak_timestep = self.data_manager.get_peak_volume_timestep(event_id)
                peak_volume_timestep = 0
                peak_volume = float('-inf')
                previous_timestep_pred = None
                for timestep, indices in timestep_dict.items():
                    #start time for this timestep
                    start_time_timestep = time.time() 
                    tiles_for_timestep = []
                    for idx in indices:
                        if self.autoregressive_model:
                            input_batch, _ = self.data_manager.get_test_batch(idx, previous_timestep_pred)
                        else:
                            input_batch, _ = self.data_manager.get_batch(idx, subset="test")
                        input_batch = input_batch.to(self.device)
                        if len(input_batch.shape) == 2:
                            input_batch = input_batch.unsqueeze(0)  # Add batch dimension
                        elif len(input_batch.shape) == 5:
                            B, T, C, H, W = input_batch.shape
                            input_batch = input_batch.view(B * T, C, H, W)
                        batch_pred = self.inference(input_batch)
                        #check if the batch pred is flattened and reshape if necessary
                        if len(batch_pred.shape) == 3 and self.data_manager.patch_domain:
                            B, C, N = batch_pred.shape
                            batch_pred = batch_pred.view(B,C,self.data_manager.tile_resolution, self.data_manager.tile_resolution)
                        if len(batch_pred.shape) == 2:
                            batch_pred = batch_pred.view(-1, self.data_manager.output_shape[0], self.data_manager.output_shape[1])
                        tiles_for_timestep.append(batch_pred)
                        
                    tiles_for_timestep = torch.cat(tiles_for_timestep, dim=0)  # Combine all tile predictions for this timestep
                    # Reconstruct the full spatial map for this event and timestep
                    if self.autoregressive_model:
                        previous_timestep_pred = tiles_for_timestep.clone().detach().float()  # Detach to prevent gradients from flowing back through time
                    if self.data_manager.patch_domain:
                        model_prediction = self.data_manager.map_sampler.reconstruct_full_map(tiles_for_timestep.detach().float())
                    else:
                        model_prediction = tiles_for_timestep.detach().float().squeeze(0) # Detach to prevent gradients from flowing back through time
                    
                    end_time_timestep = time.time() 
                    #end time for this timestep
                    
                    ref_out = self.data_manager.get_flood_map(event_id, timestep).float().to(self.device)
                    timestep_time = end_time_timestep - start_time_timestep
                    loss = self.loss_fn(model_prediction, ref_out)
                    mse = self.mse_fn(model_prediction, ref_out).item()
                    if not testing_mode:
                        running_batch_loss += loss.item()
                        running_mse += mse
                        num_batches += 1
                        logger.info(f"Event {event_id} Timestep {timestep} - Loss: {loss.item()}, MSE: {mse}")
                    else:
                        rmse = np.sqrt(mse)
                        mRMSE = self.mRMSE_fn(model_prediction, ref_out)
                        # Also save residual maps for spatial error analysis
                        residual_map = (model_prediction - ref_out).cpu().numpy()
            
                        #Compute peak volume and update peak_timestep
                        #first asssign negative values to zero to avoid inflating peak volume due to large negative errors in early timesteps before the flood peak
                        model_prediction = torch.clamp(model_prediction, min=0)
                        if model_prediction.sum() > peak_volume:
                            peak_volume = model_prediction.sum()  
                            peak_volume_timestep = timestep

                        #for now just do the 
                        if timestep == true_peak_timestep:
                            residual_output_dir = os.path.join(RUN_DIR, "residual_maps", self.config.model_name, f"event_{event_id}")
                            os.makedirs(residual_output_dir, exist_ok=True)
                            residual_map_path = os.path.join(residual_output_dir, f"residual_timestep_{timestep}_{self.config.run_id}.tif")
                            with rasterio.open(residual_map_path, 'w', driver='GTiff', height=residual_map.shape[0], width=residual_map.shape[1], count=1, dtype=residual_map.dtype) as dst:
                                dst.write(residual_map, 1)  # Write the first channel of the residual map

                        threshold = 0.1
                        pred_binary = (model_prediction > threshold).float()
                        ref_binary = (ref_out > threshold).float()
                        
                        tp = ((pred_binary == 1) & (ref_binary == 1)).sum().item()
                        tn = ((pred_binary == 0) & (ref_binary == 0)).sum().item()
                        fp = ((pred_binary == 1) & (ref_binary == 0)).sum().item()
                        fn = ((pred_binary == 0) & (ref_binary == 1)).sum().item()
                        
                        hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                        f2_score = (5* tp ) / (5 * tp + fp + 4*fn) if (tp + fp + fn) > 0 else 0
                        # f3_score = (tp - fp) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                        
                        event_metrics["mse"][event_id].append(mse)
                        event_metrics["rmse"][event_id].append(rmse)
                        event_metrics["mRMSE"][event_id].append(mRMSE)
                        event_metrics["hit_rate"][event_id].append(hit_rate)
                        event_metrics["csi"][event_id].append(csi)
                        event_metrics["f2_score"][event_id].append(f2_score)
                        # event_metrics["f3_score"][event_id].append(f3_score)
                        event_metrics["timestep_time"][event_id].append(timestep_time)
                        logger.info(f"Event {event_id} Timestep {timestep} - MSE: {mse}, RMSE: {rmse}, mRMSE: {mRMSE}, Hit Rate: {hit_rate}, CSI: {csi}, F2 Score: {f2_score}")
                
                end_time_event = time.time() #end time for this event
                event_time = end_time_event - start_time_event
                event_metrics["event_time"][event_id] = event_time
                event_metrics["peak_volume_timestep"][event_id] = peak_volume_timestep
                event_metrics["true_peak_volume_timestep"][event_id] = true_peak_timestep
                event_metrics["peak_volume_timing_error"][event_id] = peak_volume_timestep - true_peak_timestep
            
            avg_loss = running_batch_loss / num_batches if num_batches > 0 else float('inf')
            avg_mse = running_mse / num_batches if num_batches > 0 else float('inf')
            logger.info(f"Average Loss: {avg_loss} MSE: {avg_mse} ") 
            if not testing_mode:
                return avg_loss
            else:
                event_metrics["average_mse"] = avg_mse
                event_metrics["average_loss"] = avg_loss
                return event_metrics

    def test_model(self):
        logger.info("Testing model")
        if self.do_profile:
            logger.info("Profiling enabled for testing")
            with self.track_peak_gpu_memory() as peak_memory:
                metrics = self.sequential_validation(self.data_manager.test_idx, testing_mode=True)
            metrics["peak_gpu_memory"] = peak_memory.peak_mb
            metrics["flops"] = self.calculate_flops()
            metrics["parameters"] = self.calculate_model_parameters()
        else:
            metrics = self.sequential_validation(self.data_manager.test_idx, testing_mode=True)
        return metrics
    
    def mse_fn(self, pred, ref_out):
        mse = torch.mean((pred - ref_out) ** 2)
        return mse
           
    def mRMSE_fn(self, pred, ref_out):
        # Define threshold for wet cells (typically > 0.1m is considered wet)
        threshold = 0.1
        # Create binary masks for true wet cells in the reference output
        ref_wet = (ref_out > threshold).float()
        sq_err = ((pred - ref_out) ** 2) * ref_wet
        n_wet = ref_wet.sum()
        mse_wet = sq_err.sum() / torch.clamp(n_wet, min=1.0)
        rmse_wet = torch.sqrt(mse_wet)
        return float(rmse_wet.item())
    
    
    def calculate_flops(self):
        #calculate FLOPS for a single forward pass
        try:
            sample_index = self.data_manager.train_idx[0]
            input_batch, _  = self.data_manager.get_batch(sample_index)
            #if sample input has a batch dimension, remove it for FLOPS calculation
            if len(input_batch.shape) == 4:
                input_sample = input_batch[0].unsqueeze(0)  # Add batch dimension back for model input
            else:
                input_sample = input_batch.unsqueeze(0)  # Add batch dimension for model input
                
            assert input_sample.shape[0] == 1, "FLOPS calculation requires a single sample input"
                
            # Create a sample input for the model
            input_batch = torch.tensor(input_sample).to(self.device)
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
        
    def calculate_model_parameters(self):
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
            return trainable_params
        except Exception as e:
            logger.error(f"Error calculating model parameters: {e}")
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
            "horizon": self.config.horizon, 
            "random_seed": self.config.random_seed,
            "sigma": self.config.args.get('sigma', None),
        }
        return hyperparameters