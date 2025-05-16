from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.datamanager.raster.raster_loader_1dcnn import CNNRasterDataManager
from modules.utils.run_util import check_device
import logging
import numpy as np
import time
from modules.lib.constants import PICNN1D_V1
from torch.utils.flop_counter import FlopCounterMode
from torch.profiler import profile, ProfilerActivity
import os
from modules.lib.constants import SIMULATION_DATA_DIR, RUN_DIR, GRAPH_OUTPUT_DIR
import rasterio 
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_writetiff

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
        
        # Initialize weights with xavier uniform (similar to random_uniform in TensorFlow)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
    
    def create_dataset(self):
        logger.info("Creating dataset")
        self.data_manager = CNNRasterDataManager(self.config.lag, self.config.batch_size)
        self.features = self.data_manager.features
        self.outputs = self.data_manager.outputs
        logger.info(f"Features: {self.features}, Outputs: {self.outputs}")
        
    def init_model(self) -> bool:
        try:
            self.create_dataset()
            self.model = PICNN1DModel(self.steps, self.features, self.outputs).to(self.device)
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
        
        start_time = time.time()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir)) as prof:
            best_val_loss = float("inf")
            epochs_no_improvement = 0
            best_model_state = None
            best_optimizer_state = None
            losses = []
            eval_losses = []
            best_epoch = 0
            for epoch in range(self.config.epochs):
                for idx, t_indices in enumerate(self.data_manager.train_idx):
                    input_batch, output_batch = self.data_manager.get_batch(t_indices)
                    if input_batch is None or output_batch is None:
                        logger.warning(f"Error getting batch {idx}, skipping")
                        continue
                    input_in_batch = input_batch.to(device)
                    output_in_batch = output_batch.to(device)
                    model.train()
                    self.optimizer.zero_grad()
                    pred = model(input_in_batch.float())
                    y_hat_t = pred
                    y_t = output_in_batch.float()
                    y_t_plus_1 = None
                    bc_t = None
                    bc_t_plus_1 = None
                    loss = self.loss_fn(y_hat_t, y_t, y_t_plus_1, bc_t, bc_t_plus_1)
                    losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                    eval_losses.append(self.loss_eval_fn(pred, output_in_batch).detach().cpu().numpy())
                    logger.info(f"Batch train loss: {losses[-1]} eval loss: {eval_losses[-1]}")
                    
                epoch_loss = np.mean(losses)
                epoch_eval_loss = np.mean(eval_losses)
                history["loss"].append(epoch_loss)
                history["eval_loss"].append(epoch_eval_loss)
                
                # Validation
                batch_val_losses = 0
                batch_eval_val_losses = 0
                for idx, t_indices in enumerate(self.data_manager.validation_idx):
                    input_batch, output_batch = self.data_manager.get_batch(t_indices)
                    if input_batch is None or output_batch is None:
                        logger.warning(f"Error getting validation batch {idx}, skipping")
                        continue
                    input_in_batch = torch.tensor(input_batch).to(device)
                    output_in_batch = torch.tensor(output_batch).to(device)
                    model.eval()
                    with torch.no_grad():
                        pred_val = model(input_in_batch.float())
                        output_in_batch = output_in_batch.float()
                        batch_val_loss = self.loss_fn(pred_val, output_in_batch).item()
                        batch_eval_val_loss = self.loss_eval_fn(pred_val, output_in_batch).detach().cpu().numpy()
                        batch_val_losses += batch_val_loss
                        batch_eval_val_losses += batch_eval_val_loss
                        logger.info(f"Batch validation loss: {batch_val_losses} eval loss: {batch_eval_val_losses}")
                        
                val_loss = batch_eval_val_losses / len(self.data_manager.validation_idx)
                eval_val_loss = batch_eval_val_losses / len(self.data_manager.validation_idx)
                history["val_loss"].append(val_loss)
                history["eval_val_loss"].append(eval_val_loss)
                logger.info(f"Epoch {epoch} loss: {np.mean(losses)} eval loss: {np.mean(eval_losses)} validation loss: {val_loss} eval validation loss: {eval_val_loss}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    best_optimizer_state = self.optimizer.state_dict().copy()
                    epochs_no_improvement = 0
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        logger.info(f"Best validation loss: {best_val_loss} at epoch {best_epoch}")
                        break
                    
            if best_model_state is not None:
                logger.info("Loading best model state")
                model.load_state_dict(best_model_state)
        
        end_time = time.time()
        self.model = model
        train_time = end_time - start_time
        logger.info(f"Training time: {train_time}")
        history["train_time"] = train_time
        logger.info("Saving model with best validation loss")
        model_file = self.save_model_checkpoint(self.config.run_id, run_dir, best_model_state, best_optimizer_state,  self.config)
        key_averages = prof.key_averages()
        analysis_results = super().profiler_analysis(key_averages)
        logger.info(f"Memory profiling results: {key_averages.table(sort_by='cuda_memory_usage', row_limit=10)}")
        logger.info(f"Profiler analysis results: {analysis_results}")
        history['memory'] = analysis_results
        return history, train_time, model_file
    
    def test_model(self):
        logger.info("Validating model")
        model = self.model
        model.eval()
        test_data_manager = CNNRasterDataManager(self.config.lag, self.config.batch_size, self.device, test_mode=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
            with torch.no_grad():
                input_data = test_data_manager.test_input.to(self.device)
                output_data = test_data_manager.test_output.to(self.device)
                start_time = time.time()
                pred = self.model(input_data)
                end_time = time.time()
                
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
            # Convert tensor to Python scalar
            nse = self.nse_fn(observed, predicted)
            pred_time = end_time - start_time
            logger.info(f"Validation prediction_time:{pred_time} loss MSE: {mse} RMSE: {rmse} eval loss L1: {eval_loss} NSE: {nse}")
            flops = self.calculate_flops()
            self.save_predictions(input_data, pred)
            wet_acc,wet_precision, wet_recall, wet_f1 = self.wet_cell_classification_accuracy(pred, ref_out)
            rmse_wet = self.wet_cell_depth_rmse(pred, ref_out)
            analysis_results = super().profiler_analysis(prof.key_averages())
            metrics = {
                "mse": mse,
                "rmse": rmse,
                "nse": nse,
                "pred_time": pred_time,
                "flops": flops,
                "wet_rmse": rmse_wet,
                "wet_acc": wet_acc,
                "wet_precision": wet_precision, 
                "wet_recall": wet_recall,
                "wet_f1": wet_f1,
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
    
    def wet_cell_depth_rmse(self, pred, ref_out):
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
    
    def wet_cell_classification_accuracy(self, pred, ref_out):
        # Define threshold for wet cells (typically > 0.01m is considered wet)
        threshold = 0.01
        
        # Create binary masks
        pred_wet = (pred > threshold).float()
        ref_wet = (ref_out > threshold).float()
        
        # True positives: cells correctly predicted as wet
        true_positives = torch.sum((pred_wet == 1) & (ref_wet == 1)).float()
        
        # False positives: cells incorrectly predicted as wet
        false_positives = torch.sum((pred_wet == 1) & (ref_wet == 0)).float()
        
        # False negatives: wet cells incorrectly predicted as dry
        false_negatives = torch.sum((pred_wet == 0) & (ref_wet == 1)).float()
        
        # True negatives: correctly predicted dry cells
        true_negatives = torch.sum((pred_wet == 0) & (ref_wet == 0)).float()
        
        # Calculate metrics
        total = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Convert to Python scalars
        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'true_positives': true_positives.item(),
            'false_positives': false_positives.item(),
            'false_negatives': false_negatives.item(),
            'true_negatives': true_negatives.item()
        }
        
        logger.info(f"Wet cell classification - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")
        return metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']
    
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
        
            # Use FlopCounterMode to count FLOPS
            with FlopCounterMode(self.model) as counter:
                _ = self.model(sample_input)
                
            flops = counter.get_total_flops()
            logger.info(f"FLOPS: {flops}")
            flops_str = self.format_flops(flops)
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
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None
        
    def loss_fn(self, y_hat_t, y_t, y_t_plus_1, bc_t, bc_t_plus_1):
        # Need to implement the physics informed loss function
        # Equation 1: Loss = MSE + physics informed loss
        delta_x = 5
        delta_y = 5
        n_x = 911
        n_y = 651
        delta_t_val = 15 * 60 #15 minutes in seconds
        area = delta_x * delta_y * n_x * n_y
        
        v_t = self.calculate_v_t(delta_x, delta_y, y_t)
        v_t_plus_1 = self.calculate_v_t(delta_x, delta_y, y_t_plus_1, n_x, n_y)
        v_hat_t = self.calculate_v_hat_t(delta_x, delta_y, y_hat_t, n_x, n_y)
        mse_loss = self.mse(y_t, y_hat_t)
        loss_calculated = self.calculate_loss(v_hat_t, v_t, v_t_plus_1, delta_t_val, bc_t, bc_t_plus_1, area, mse_loss)
        print(f"Calculated Loss: {loss_calculated}")

    def relu(self, x):
        return nn.ReLU()(x)

    def calculate_v_t(self, delta_x, delta_y, y_t):
        return delta_x * delta_y * np.sum(y_t)

    def calculate_v_hat_t(self, delta_x, delta_y, y_hat_t):
        return delta_x * delta_y * np.sum(y_hat_t)

    def mse(self, y_t, y_hat_t):
        return nn.MSELoss()(y_hat_t, y_t) 

    def calculate_loss(self, v_hat_t, v_t, v_t_plus_1, delta_t, bc_t, bc_t_plus_1, area, mse_loss):
        relu_arg1 = v_hat_t - v_t - delta_t * np.sum(bc_t)
        term2 = (self.relu(relu_arg1)**2) / (area)
        relu_arg2 = v_t_plus_1 - v_hat_t - delta_t * np.sum(bc_t_plus_1)
        term3 = (self.elu(relu_arg2)**2) / (area)
        return mse_loss + term2 + term3