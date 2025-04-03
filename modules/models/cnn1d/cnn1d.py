from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.dataloader.raster.raster_loader_1dcnn import CNNRasterDataManager
from modules.utils.run_util import check_device
import logging
import numpy as np
import time
from modules.lib.constants import CNN1D_V1
from torch.utils.flop_counter import FlopCounterMode
import os
from modules.lib.constants import CARLISLE_DATA_DIR, SIMULATION_DATA_DIR, RUN_DIR
import rasterio 
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_writetiff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("1DCNN_ModelWrapper")

model_name  = CNN1D_V1

class CNNModel(nn.Module):
    def __init__(self, steps, features, outputs):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * steps, 32)
        self.fc2 = nn.Linear(32, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, outputs)
        self.relu = nn.ReLU()
    
    def forward(self, x, dw_filter, ref, trainning_with_dw_mask=True):
        # In PyTorch, Conv1d expects [batch, channels, length]
        # We need to transpose from [batch, length, channels]
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
            
class CNN1DSAModelWrapper(ModelWrapper):
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        self.device = check_device()
        self.lag = config.lag
        self.steps = 1
        self.features = self.lag * 3
        self.outputs = 581067
    
    def create_dataset(self):
        logger.info("Creating dataset")
        self.data_manager = CNNRasterDataManager(self.config.lag, self.config.batch_size)
        self.features = self.data_manager.features
        self.outputs = self.data_manager.outputs
        
    def init_model(self) -> bool:
        try:
            self.create_dataset()
            self.model = CNNModel(self.steps, self.features, self.outputs).to(self.device)
            self.loss_fn= nn.MSELoss()
            self.loss_eval_fn = nn.L1Loss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            return True
        
        except Exception as e:
            logger.error(f"Error creating 1DCNN model: {e}")
            return False
        
    def train(self, run_dir: str):
        model = self.model
        device = self.device
        history = {
            "loss": [],
            "eval_loss": [],
            "val_loss": [],
            "eval_val_loss": [],
            "train_time": None
        }
        self.log_memory_stats(phase="startup")
        start_time = time.time()
        for epoch in range(self.config.epochs):
            epoch_start_mem = self.log_memory_stats(phase="epoch_start", epoch=epoch)
            best_val_loss = float("inf")
            epochs_no_improvement = 0
            best_model_state = None
            losses = []
            eval_losses = []
            for idx, t_indices in enumerate(self.data_manager.train_idx):
                input_batch, output_batch = self.data_manager.get_batch(t_indices)
                if input_batch is None or output_batch is None:
                    logger.warning(f"Error getting batch {idx}, skipping")
                    continue
                input_in_batch = input_batch.to(device)
                output_in_batch = output_batch.to(device)
                model.train()
                self.optimizer.zero_grad()
                pred = model(input_in_batch.float(), None, output_in_batch.float(), False)
                output_in_batch = output_in_batch.float()
                loss = self.loss_fn(pred, output_in_batch)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                eval_losses.append(self.loss_eval_fn(pred, output_in_batch).detach().cpu().numpy())
                logger.info(f"Batch train loss: {losses[-1]} eval loss: {eval_losses[-1]}")
                
            train_end_mem = self.log_memory_stats(phase="train_complete", epoch=epoch)
            epoch_loss = np.mean(losses)
            epoch_eval_loss = np.mean(eval_losses)
            history["loss"].append(epoch_loss)
            history["eval_loss"].append(epoch_eval_loss)
            val_losses = []
            eval_val_losses = []
            for idx, t_indices in enumerate(self.data_manager.validation_idx):
                input_batch, output_batch = self.data_manager.get_batch(t_indices)
                if input_batch is None or output_batch is None:
                    logger.warning(f"Error getting validation batch {idx}, skipping")
                    continue
                input_in_batch = torch.tensor(input_batch).to(device)
                output_in_batch = torch.tensor(output_batch).to(device)
                model.eval()
                with torch.no_grad():
                    pred_val = model(input_in_batch.float(), None, output_in_batch.float(), False)
                    output_in_batch = output_in_batch.float()
                    batch_val_loss = self.loss_fn(pred_val, output_in_batch).item()
                    batch_eval_val_loss = self.loss_eval_fn(pred_val, output_in_batch).detach().cpu().numpy()
                    val_losses.append(batch_val_loss)
                    eval_val_losses.append(batch_eval_val_loss)
                    logger.info(f"Batch validation loss: {batch_val_loss} eval loss: {batch_eval_val_loss}")
            mean_val_loss = np.mean(val_losses)
            mean_eval_val_loss = np.mean(eval_val_losses)
            history["val_loss"].append(mean_val_loss)
            history["eval_val_loss"].append(mean_eval_val_loss)
            logger.info(f"Epoch {epoch} loss: {np.mean(losses)} eval loss: {np.mean(eval_losses)} validation loss: {mean_val_loss} eval validation loss: {mean_eval_val_loss}")
            
            if mean_eval_val_loss < best_val_loss:
                best_val_loss = mean_eval_val_loss
                best_model_state = model.state_dict().copy()
                epochs_no_improvement = 0
                logger.info(f"Saving model with best validation loss: {best_val_loss}")
                self.save_model_checkpoint(self.config.run_id, run_dir, self.model, self.config)
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch} with validation loss: {best_val_loss}")
                    break
                
        if best_model_state is not None:
            logger.info("Loading best model state")
            model.load_state_dict(best_model_state)
    
        final_mem = self.log_memory_stats(phase="training_complete")
        end_time = time.time()
        self.model = model
        train_time = end_time - start_time
        logger.info(f"Training time: {train_time}")
        history["train_time"] = train_time
        model_file = self.save_model_checkpoint(self.config.run_id, run_dir, self.model, self.config)
        return history, train_time, model_file
        
    def validate_model(self):
        logger.info("Validating model")
        model = self.model
        self.log_memory_stats(phase="inference_start")
        model.eval()
        test_data_manager = CNNRasterDataManager(self.config.lag, self.config.batch_size, self.device, test_mode=True)
        with torch.no_grad():
            input_data = torch.tensor(test_data_manager.input_data).to(self.device)
            output_data = torch.tensor(test_data_manager.test_output).to(self.device)
            self.log_memory_stats(phase="inference_data_loaded")
            start_time = time.time()
            pred = self.model(input_data, None, output_data, False)
            end_time = time.time()
            self.log_memory_stats(phase="inference_complete")
            
        ref_out = output_data.float()
        loss = self.loss_fn(pred, ref_out)
        mse = loss.item()
        rmse = np.sqrt(mse)
        
        # Calculate RMSE for wet cells only (where reference > 0)
        wet_mask = (ref_out > 0)
        if torch.sum(wet_mask) > 0:  # Check if there are any wet cells
            pred_wet = pred[wet_mask]
            ref_wet = ref_out[wet_mask]
            wet_loss = self.loss_fn(pred_wet, ref_wet)
            rmse_wet = np.sqrt(wet_loss.item())
            logger.info(f"Wet cells RMSE: {rmse_wet}")
        else:
            rmse_wet = None
            logger.warning("No wet cells found in reference data")
        
        eval_loss = self.loss_eval_fn(pred, ref_out).detach().cpu().numpy()
        # Calculate Nash-Sutcliffe Efficiency (NSE) using PyTorch operations
        observed = ref_out
        predicted = pred
        observed_mean = torch.mean(observed)
        numerator = torch.sum((observed - predicted) ** 2)
        denominator = torch.sum((observed - observed_mean) ** 2)
        nse = 1 - (numerator / denominator)
        nse = nse.item()  # Convert tensor to Python scalar
        
        logger.info(f"Validation loss MSE: {mse} RMSE: {rmse} eval loss L1: {eval_loss} NSE: {nse}")
        pred_time = end_time - start_time
        logger.info("Prediction time: " + str(pred_time))
        
        flops = self.calculate_flops()
        self.save_predictions(input_data, pred)
        wet_cell_accuracy = self.wet_cell_classification_accuracy(pred, ref_out)
        return mse, rmse, nse, pred_time, flops, rmse_wet
    
    def wet_cell_classification_accuracy(self, pred, ref_out):
        return 0
        
    def save_predictions(self, input_data, pred):
        try:
            # Create output directory for prediction maps for maximum extent
            output_dir = os.path.join(self.config.run_dir, "output_maps")
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Saving prediction maps to {output_dir}")
            
            # Get reference raster for metadata
            ref_file = os.path.join(SIMULATION_DATA_DIR, "Run1-0000.wd")
            
            # Get dimensions from reference file
            with rasterio.open(ref_file) as src:
                height = src.height
                width = src.width
                profile = src.profile
                transform = src.transform
                crs = src.crs
            
            logger.info(f'Prediction shape: {pred.shape}, Reshaping to dimensions: {height}x{width}')
            
            # We'll only save the prediction for maximum flood extent (index 145)
            idx = 145
            pred_max = pred.detach().cpu().numpy()[idx]
            
            # Reshape the 1D prediction array to 2D using proper dimensions
            pred_reshaped = pred_max.reshape(height, width)
            
            # Define output file path
            out_file = os.path.join(output_dir, f"map_{idx:04d}.wd")
            
            # Create a copy of the profile for the output file
            out_profile = profile.copy()
            out_profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw'
            )
            
            # Write the reshaped prediction directly to a .wd file
            with rasterio.open(out_file, 'w', **out_profile) as dst:
                dst.write(pred_reshaped.astype(rasterio.float32), 1)
            
            logger.info(f"Saved prediction map {idx} to {out_file}")
            return output_dir
        
        except Exception as e:
            logger.error(f"Error saving prediction maps: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def calculate_flops(self):
        try:
            t_indices = self.data_manager.train_idx[0]
            input_batch, _  = self.data_manager.get_batch(t_indices)

            # Create a sample input for the model
            input_batch = torch.tensor(input_batch).to(self.device)
            sample_input = input_batch[0].unsqueeze(0)
        
            # Use FlopCounterMode to count FLOPs
            with FlopCounterMode(self.model) as counter:
                _ = self.model(sample_input, None, None, False)
            
            flops = counter.get_total_flops()
            flops_str = self.format_flops(flops)
            logger.info(f"Model FLOPS: {flops_str}")
            return flops
        except Exception as e:
            logger.error(f"Error calculating FLOPS: {e}")
            return None, None
    
    def format_flops(self, flops):
        """Convert FLOPS to a human-readable string."""
        if flops < 1e9:
            return f"{flops / 1e6:.2f} MFLOPS"
        else:
            return f"{flops / 1e9:.2f} GFLOPS"

    def save_model_checkpoint(self, run_id, run_dir, model, config):
        try:
            model_file = os.path.join(run_dir, f"{config.model_name}_{run_id}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.epochs,
                'run_id': run_id,
            }, os.path.join(run_dir, model_file))
            return model_file
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None