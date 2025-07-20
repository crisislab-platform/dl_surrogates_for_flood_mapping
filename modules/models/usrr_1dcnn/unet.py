from modules.models.model_wrapper import ModelWrapper
from modules.models.model_wrapper import ModelConfig
from modules.datamanager.raster.raster_loader_unet import UNetDataManager
from modules.utils.run_util import check_device
from modules.lib.constants import USRR_UNET_V1
from torch.profiler import profile, ProfilerActivity
from modules.utils.model_util import profiler_analysis

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import os

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
        self.data_manager = None
    
    def create_dataset(self):
        self.data_manager = UNetDataManager(self.sampling_dist,  self.config.run_dir, batches_per_map = self.config.batch_size)
        return True
        
    def init_model(self):
        self.create_dataset()
        device = check_device()
        # Define the model structure by the number of channels in each layer. input channel is 2 (x, y) and output channel is 1 (target).
        self.model_structure = [2, 32, 64, 128, 256] 
        self.model = UNet(encoder_channels= self.model_structure, decoder_channels= self.model_structure[:0:-1]).to(device)
        self.model.float() 
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        logger.info(f"UNet model created")
        return True
    
    def train_model(self, run_dir, tuning_mode=True):
        return super().train_model(run_dir, tuning_mode)
    
    def train(self, run_dir: str, tuning_mode = True):
        return super().train(run_dir, tuning_mode)
        
    def test_model(self):
        self.model.eval()
        batch_loss = 0
        batch_mRMSE = 0
        batch_nse = 0
        start_time = time.time()
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.run_dir)) as prof:
            with torch.no_grad():
                for idx, batch_list in enumerate(self.data_manager.test_input_batches): 
                    output_batch_list = self.data_manager.test_output_batches[idx]
                    
                    # Process each individual tensor in the batch list
                    batch_preds = []
                    batch_losses = []
                    batch_mrmses = []
                    batch_nses = []
                    
                    for i, input_tensor in enumerate(batch_list):
                        output_tensor = output_batch_list[i]
                        # Ensure input is a proper tensor with batch dimension
                        if not isinstance(input_tensor, torch.Tensor):
                            logger.error(f"Expected tensor, got {type(input_tensor)}")
                            continue
                            
                        # Process a single tensor through the model
                        pred = self.model(input_tensor)
                        loss = self.loss_fn(pred, output_tensor)
                        batch_losses.append(loss.item())
                        mRMSE = self.mRMSE_fn(pred, output_tensor)
                        nse = self.nse_fn(output_tensor, pred)
                        batch_mrmses.append(mRMSE)
                        batch_nses.append(nse)
                        batch_preds.append(pred)
                    
                    # Compute average metrics for this batch
                    avg_loss = np.mean(batch_losses) if batch_losses else 0
                    avg_mRMSE = np.mean(batch_mrmses) if batch_mrmses else 0
                    avg_nse = np.mean(batch_nses) if batch_nses else 0
                    
                    batch_loss += avg_loss
                    batch_mRMSE += avg_mRMSE
                    batch_nse += avg_nse
                    prof.step()
                    logger.info(f"Test Loss for batch {idx}: {avg_loss} mRMSE: {avg_mRMSE} NSE: {avg_nse}")
                
                end_time = time.time()
                # Calculate final metrics
                num_batches = len(self.data_manager.test_input_batches)
                mse = batch_loss / num_batches if num_batches > 0 else 0
                rmse = np.sqrt(mse)
                mRMSE = batch_mRMSE / num_batches if num_batches > 0 else 0
                nse = batch_nse / num_batches if num_batches > 0 else 0
                
                logger.info(f"Test Loss : {mse} mRMSE: {mRMSE} NSE: {nse} RMSE: {rmse}")
                
        # Calculate NSEß
        key_averages = prof.key_averages()
        pred_time = end_time - start_time
        logger.info(f"Validation prediction_time:{pred_time} loss MSE: {mse} RMSE: {rmse}  NSE: {nse} mRMSE: {mRMSE}")
        flops = self.calculate_flops()
        
        analysis_results = profiler_analysis(key_averages)
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


    def save_model_checkpoint(self, run_id, run_dir, model_state, config):
        try:
            model_file = os.path.join(run_dir, f"{config.model_name}_{run_id}.pth")
            torch.save({
                'model_state_dict': model_state,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.epochs,
                'run_id': run_id,
                'model_structure': self.model_structure,
                'sampling_dist': self.sampling_dist
            }, model_file) 
            return model_file      
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None
            
    def create_hyperparameters_dict(self):
        return {
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "sampling_dist": self.sampling_dist,
            "model_name": self.model_name,
            "run_id": self.config.run_id
        }
