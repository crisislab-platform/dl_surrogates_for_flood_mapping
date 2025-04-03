from modules.models.model_wrapper import ModelWrapper
from modules.models.model_wrapper import ModelConfig
from modules.dataloader.raster.raster_loader_unet import UNetDataManager
from modules.utils.run_util import check_device
from modules.lib.constants import USRR_UNET_V1

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
import logging
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USSR_UNET_Model")

model_name =  USRR_UNET_V1

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)     # yz: added padding to retain dimension
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)     # yz: added padding to retain dimension
        self.activation_f = nn.ReLU()

    def forward(self, x):
        return self.activation_f(self.conv2(self.activation_f(self.conv1(x))))

class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]   # yz: ensured size is the same with x, no crop needed
            # enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs, dec_chs, num_class=1,
                 retain_dim=False, out_sz=(64, 64)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
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
        model = UNet(enc_chs= self.model_structure, dec_chs= self.model_structure[:0:-1]).to(device)
        model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.eval_loss_fn = nn.L1Loss()
        self.model = model
        logger.info(f"UNet model created")
        return True
        
    def train(self, run_dir: str):
        
        losses = []
        eval_losses = []
        val_losses = []
        eval_val_losses = []
        
        history = {
            "loss": [],
            "eval_loss": [],
            "val_loss": [],
            "eval_val_loss": [], 
            "train_time": None,
        }
       
        start_time = time.time()
        model = self.model
        for epoch in range(1,self.config.epochs+1):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            for idx in self.data_loader.train_idxs:
                x, y = self.data_loader.get_batch(idx)
                if x.shape[0] == 0:
                    logger.error(f"Input empy skipping batch {idx}")
                    continue
                model.train() # set model to training mode
                optimizer = self.optimizer
                optimizer.zero_grad()
                pred = model(x.float())
                loss = self.loss_fn(pred, y.float())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                eval_loss = self.eval_loss_fn(pred, y.float())
                eval_losses.append(self.eval_loss_fn(pred, y.float()).detach().cpu().numpy())
                
                #check if there is any NaN in the loss
                if torch.isnan(loss):
                    logger.error("Loss is NaN, stopping training")
                    exit(1)
                logger.info(f"Epoch {epoch}, Batch {idx} - Loss: {loss.item()} Eval loss: {eval_loss.item()}")
            with torch.no_grad():
                for val_idx in self.data_loader.val_idxs:
                    x_val, y_val = self.data_loader.get_batch(val_idx)
                    if x_val.shape[0] == 0:
                        logger.error(f"Input empy skipping batch {val_idx}")
                        continue
                    model.eval()
                    pred_val = model(x_val.float())
                    output_ref = y_val.float()
                    val_loss = self.loss_fn(pred_val, output_ref)
                    val_losses.append(val_loss.item())
                    eval_val_loss = self.eval_loss_fn(pred_val, output_ref).detach().cpu().numpy()
                    eval_val_losses.append(eval_val_loss)
                    
            train_loss = np.mean(losses[-len(self.data_loader.train_idxs):])
            val_loss = np.mean(val_losses[-len(self.data_loader.val_idxs):])
            eval_loss = np.mean(eval_losses[-len(self.data_loader.train_idxs):])
            eval_val_loss = np.mean(eval_val_losses[-len(self.data_loader.val_idxs):])
            
            logger.info(f"Epoch: {epoch}")
            logger.info(f"Train loss: {train_loss}")
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(f"Validation loss: {val_loss}")
            logger.info(f"Eval validation loss: {eval_val_loss}")
            
            history["loss"].append(train_loss)
            history["eval_loss"].append(eval_loss)
            history["val_loss"].append(val_loss)
            history["eval_val_loss"].append(eval_val_loss)
            
            # Add early stopping condition
            if epoch > 5 and np.mean(history["val_loss"][-len(self.data_loader.val_idxs):]) < 0.01:
                logger.info("Early stopping condition met")
                self.model = model
                break
        
        self.model = model
        end_time = time.time()
        train_time = end_time - start_time
        history["train_time"] = train_time
        logger.info(f"Training completed in {train_time} seconds")
        model_file = self.save_model_checkpoint(self.config.run_id, run_dir, self.model, self.config)
        return history, train_time, model_file
    
    def predict(self):
   
        device = check_device()
        self.model.eval()
        losses = []
        with torch.no_grad():
            start_time = time.time()
            logger.info("Running inference on test data")
            for idx in self.data_loader.test_idxs:
                x_test, y_test = self.data_loader.get_batch(idx)
                logger.info(f"Test data shape: {x_test.shape}")
                if x_test.shape[0] == 0:
                    logger.error(f"Input empy skipping batch {idx}")
                    continue
                x_test = x_test.float().to(device)
                pred = self.model(x_test)
                loss = self.loss_fn(pred, y_test.float())
                logger.info(f"Test Loss: {loss.item()}")
                losses.append(loss.item())
            mse = np.mean(losses)
            rmse = np.sqrt(mse)
        end_time = time.time()
        pred_time = end_time - start_time
        return mse, rmse, pred_time
    
    def save_model_checkpoint(self, run_id, run_dir, model, config):
        try:
            model_file = os.path.join(run_dir, f"{model_name}_{run_id}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_structure': self.model_structure,
                'sampling_dist': self.sampling_dist,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.data_loader.batch_size,
                'model_strcture': self.model_structure,
                'num_epochs': self.config.epochs,
                'run_id': run_id,
            }, os.path.join(run_dir, model_file))
            return model_file
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
        return None
    
    def validate_model(self):
        mse, rmse, pred_time = self.predict()
        return mse, rmse, pred_time