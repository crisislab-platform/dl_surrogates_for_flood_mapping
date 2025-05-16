from modules.models.model_wrapper import ModelWrapper
from modules.models.model_wrapper import ModelConfig
from modules.datamanager.raster.raster_loader_unet import UNetDataManager
from modules.utils.run_util import check_device
from modules.lib.constants import USRR_UNET_V1, RUN_DIR
from torch.utils.flop_counter import FlopCounterMode
from torch.profiler import profile, ProfilerActivity

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USSR_UNET_Model")

model_name =  USRR_UNET_V1

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)   
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.activation_f = nn.ReLU()

    def forward(self, x):
        return self.activation_f(self.conv2(self.activation_f(self.conv1(x))))

class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.encoder_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
        self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_class=1,
                 retain_dim=False, output_size=(64, 64)):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.output_size = output_size

    def forward(self, x):
        encoded_features = self.encoder(x)
        out = self.decoder(encoded_features[::-1][0], encoded_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.output_size)
        return out

class UNetModelWrapper(ModelWrapper):
    def __init__(self , config: ModelConfig):
        super().__init__(config)
        self.model_name = USRR_UNET_V1
        self.sampling_dist = config.args.get('sampling_dist', 300)
        self.data_loader = None
        pass
    
    def create_dataset(self):
        self.data_loader = UNetDataManager(self.sampling_dist, self.config.run_dir, self.config.epochs)
        return True
        
    def init_model(self):
        self.create_dataset()
        device = check_device()
        self.model_structure = [2, 32, 64, 128, 256]
        model = UNet(encoder_channels= self.model_structure, decoder_channels= self.model_structure[:0:-1]).to(device)
        model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.eval_loss_fn = nn.L1Loss()
        self.model = model
        logger.info(f"UNet model created")
        return True
    
    def train(self, run_dir: str, tuning_mode = True):
        super().train(run_dir, tuning_mode)
        
    def test_model(self):
        super().test_model()
        
    # def train(self, run_dir: str):
    #     losses = []
    #     eval_losses = []
    #     val_losses = []
    #     eval_val_losses = []
        
    #     history = {
    #         "loss": [],
    #         "eval_loss": [],
    #         "val_loss": [],
    #         "eval_val_loss": [], 
    #         "train_time": None,
    #     }

    #     best_val_loss = float('inf')
    #     best_epoch = 0
    #     best_model_state = None
    #     epochs_no_improvement = 0
   
    #     start_time = time.time()
    #     model = self.model
        
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(run_dir)) as prof:
    #         for epoch in range(1,self.config.epochs+1):
    #             model.train()
    #             logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
    #             for idx in self.data_loader.train_idxs:
    #                 x, y = self.data_loader.get_batch(idx)
    #                 if x.shape[0] == 0:
    #                     logger.error(f"Input empy skipping batch {idx}")
    #                     continue
    #                 model.train() # set model to training mode
    #                 optimizer = self.optimizer
    #                 optimizer.zero_grad()
    #                 pred = model(x.float())
    #                 loss = self.loss_fn(pred, y.float())
    #                 losses.append(loss.item())
    #                 loss.backward()
    #                 optimizer.step()
    #                 eval_loss = self.eval_loss_fn(pred, y.float())
    #                 eval_losses.append(self.eval_loss_fn(pred, y.float()).detach().cpu().numpy())
                    
    #                 if torch.isnan(loss):
    #                     logger.error("Loss is NaN, stopping training")
    #                     exit(1)
    #                 logger.info(f"Epoch {epoch}, Batch {idx} - Loss: {loss.item()} Eval loss: {eval_loss.item()}")
    #             with torch.no_grad():
    #                 for val_idx in self.data_loader.val_idxs:
    #                     x_val, y_val = self.data_loader.get_batch(val_idx)
    #                     if x_val.shape[0] == 0:
    #                         logger.error(f"Input empy skipping batch {val_idx}")
    #                         continue
    #                     model.eval()
    #                     pred_val = model(x_val.float())
    #                     output_ref = y_val.float()
    #                     val_loss = self.loss_fn(pred_val, output_ref)
    #                     val_losses.append(val_loss.item())
    #                     eval_val_loss = self.eval_loss_fn(pred_val, output_ref).detach().cpu().numpy()
    #                     eval_val_losses.append(eval_val_loss)
                        
    #             train_loss = np.mean(losses[-len(self.data_loader.train_idxs):])
    #             val_loss = np.mean(val_losses[-len(self.data_loader.val_idxs):])
    #             eval_loss = np.mean(eval_losses[-len(self.data_loader.train_idxs):])
    #             eval_val_loss = np.mean(eval_val_losses[-len(self.data_loader.val_idxs):])
                
    #             logger.info(f"Epoch: {epoch}")
    #             logger.info(f"Train loss: {train_loss}")
    #             logger.info(f"Eval loss: {eval_loss}")
    #             logger.info(f"Validation loss: {val_loss}")
    #             logger.info(f"Eval validation loss: {eval_val_loss}")
                
    #             history["loss"].append(train_loss)
    #             history["eval_loss"].append(eval_loss)
    #             history["val_loss"].append(val_loss)
    #             history["eval_val_loss"].append(eval_val_loss)
                
    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss
    #                 best_model_state = model.state_dict().copy()
    #                 epochs_no_improvement = 0
    #                 best_epoch = epoch
    #                 logger.info(f"Saving model with best validation loss: {best_val_loss}")
    #                 self.save_model_checkpoint(self.config.run_id, run_dir, self.model, self.config)
    #             else:
    #                 epochs_no_improvement += 1
    #                 if epochs_no_improvement >= self.config.patience:
    #                     logger.info(f"Early stopping at epoch {epoch} with validation loss: {best_val_loss}")
    #                     break
        
    #     if best_model_state is not None:
    #         logger.info(f"Restoring best model state from epoch {best_epoch} with validation loss: {best_val_loss}")
    #         model.load_state_dict(best_model_state)
        
    #     self.model = model
    #     end_time = time.time()
    #     train_time = end_time - start_time
    #     history["train_time"] = train_time
    #     logger.info(f"Training completed in {train_time} seconds")
        
    #     key_averages = prof.key_averages()
    #     analysis_results = super().profiler_analysis(key_averages)
    #     logger.info(f"Memory profiling results: {key_averages.table(sort_by='cuda_memory_usage', row_limit=10)}")
    #     logger.info(f"Profiler analysis results: {analysis_results}")
    #     history['memory'] = analysis_results
        
    #     model_file = self.save_model_checkpoint(self.config.run_id, run_dir, self.model, self.config)
    #     return history, train_time, model_file
    
    # def test_model(self):
    #     device = check_device()
    #     self.model.eval()
    #     losses = []
    #     nses = []
        
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
    #         with torch.no_grad():
    #             start_time = time.time()
    #             logger.info("Running inference on test data")
    #             for idx in self.data_loader.test_idxs:
    #                 x_test, y_test = self.data_loader.get_batch(idx)
    #                 logger.info(f"Test data shape: {x_test.shape}")
    #                 if x_test.shape[0] == 0:
    #                     logger.error(f"Input empy skipping batch {idx}")
    #                     continue
    #                 x_test = x_test.float().to(device)
    #                 pred = self.model(x_test)
    #                 loss = self.loss_fn(pred, y_test.float())
    #                 nse = self.nse_fn(pred, y_test.float())
    #                 logger.info(f"Test Loss: {loss.item()}")
    #                 losses.append(loss.item())
    #             mse = np.mean(losses)
    #             rmse = np.sqrt(mse)
    #             nse = np.mean(nses)
    #             flops = self.calculate_flops()
    #             wet_rmse = None
    #             wet_acc = None
    #         end_time = time.time()
    #         pred_time = end_time - start_time
        
    #     key_averages = prof.key_averages()
    #     analysis_results = super().profiler_analysis(key_averages)
    #     logger.info(f"Validation profiling results: {key_averages.table(sort_by='cuda_memory_usage', row_limit=10)}")
        
    #     metrics = {
    #         'mse': mse,
    #         'rmse': rmse,
    #         'nse': nse,
    #         'pred_time': pred_time,
    #         'wet_rmse': wet_rmse,
    #         'wet_acc': wet_acc,
    #         'flops': flops,
    #         "pred_memory_usage": analysis_results
    #     }
    #     return metrics
    
    # def calculate_flops(self):
    #     idx = self.data_loader.test_idxs[0]
    #     x, y = self.data_loader.get_batch(idx)
    #     x = x[0:1]
    #     self.model.eval()
        
    #     with FlopCounterMode(self.model) as counter:
    #         _ = self.model(x.float())
    #         flops = counter.get_total_flops()
    #         flops_str = self.format_flops(flops)
    #         logger.info(f"Model FLOPS: {flops_str}")
    #         return flops

    # def save_model_checkpoint(self, run_id, run_dir, model, config):
    #     try:
    #         model_file = os.path.join(run_dir, f"{model_name}_{run_id}.pt")
    #         metadata_dir = os.path.join(RUN_DIR, self.config.model_name)
    #         metadata_file = os.path.join(metadata_dir, "run_metadata.csv")
        
    #         torch.save({
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': self.optimizer.state_dict(),
    #             'model_structure': self.model_structure,
    #             'sampling_dist': self.sampling_dist,
    #             'learning_rate': self.config.learning_rate,
    #             'batch_size': self.data_loader.batch_size,
    #             'model_strcture': self.model_structure,
    #             'num_epochs': self.config.epochs,
    #             'run_id': run_id,
    #         }, os.path.join(run_dir, model_file))
            
    #         os.makedirs(metadata_dir, exist_ok=True)
            
    #         if os.path.exists(metadata_file):
    #             logger.info(f"Metadata file exists: {metadata_file}")
    #             run_metadata = pd.read_csv(metadata_file)
    #         else:
    #             logger.info(f"Creating new metadata file: {metadata_file}")
    #             run_metadata = pd.DataFrame(columns=["run_id", "sampling_dist", "model_name", "model_file"])
            
    #         new_row = pd.DataFrame({
    #             "run_id": [run_id],
    #             "sampling_dist": [self.sampling_dist],
    #             "model_name": [self.model_name],
    #             "model_file": [model_file]
    #         })
            
    #         run_metadata = pd.concat([run_metadata, new_row], ignore_index=True)
    #         run_metadata.to_csv(metadata_file, index=False)  
    #         logger.info(f"Run metadata saved to {metadata_file}")
    #         return model_file
        
    #     except Exception as e:
    #         logger.error(f"Error saving model metrics: {e}")
    #     return None