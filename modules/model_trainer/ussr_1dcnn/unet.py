from modules.model_trainer.model import Model
from modules.model_trainer.model import ModelConfig
from modules.dataloader.raster.raster_loader_unet import UNetDataManager
from modules.utils.run_util import check_device
from modules.utils.path_util import RUN_DIR
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.transforms import CenterCrop
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USSR_1D_CNN_Model")

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

class UNetModelWrapper(Model):
    def __init__(self , config: ModelConfig):
        super().__init__(config)
        self.sampling_dist = config.args.get('sampling_dist', 300)
        self.data_loader = None
        pass
    
    def create_dataset(self):
        self.data_loader = UNetDataManager(self.sampling_dist, self.config.run_dir, self.config.epochs)
        return True
        
    def init_model(self):
        self.create_dataset()
        device = check_device()
        model_structure = [2, 32, 64, 128, 256]
        model = UNet(enc_chs= model_structure, dec_chs= model_structure[:0:-1]).to(device)
        model.float()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.eval_loss_fn = nn.L1Loss()
        self.model = model
        logger.info(f"UNet model created")
        
    def train(self, run_dir: str):
        history = {
            "loss": [],
            "eval_loss": [],
            "val_loss": []
        }
        start_time = time.time()
        for epoch in range(self.config.epochs):
            for idx in self.data_loader.train_idxs:
                x, y = self.data_loader.get_batch(idx)
                model = self.model
                model.train() # set model to training mode
                optimizer = self.optimizer
                optimizer.zero_grad()
                pred = model(x.float())
                loss = self.loss_fn(pred, y.float())
                history["losses"].append(loss.detach().cpu().numpy())
                loss.backward()
                optimizer.step()
                history["eval_losses"].append(self.eval_loss_fn(pred, y.float()).detach().cpu().numpy())
                logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")
        end_time = time.time()
        train_time = end_time - start_time
        logger.info(f"Training completed in {train_time} seconds")
        return history, train_time
    
    def predict(self):
        pass
    