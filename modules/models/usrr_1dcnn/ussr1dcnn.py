from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.models.usrr_1dcnn.unet import UNetModelWrapper
from modules.models.usrr_1dcnn.cnn1d import CNN1DModelWrapper
from modules.models.usrr_1dcnn.spatial_reduction_module.rl_culster_finder import RLClusterFinder
from modules.models.usrr_1dcnn.spatial_reduction_module.reconstruction import validate_reconstruction
import logging
import concurrent.futures
import time
import os
import glob
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USSR_1D_CNN_Model")

class USSR1DCNNModelConfig(ModelConfig):
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    model_name: str
    dropout_rate: float = 0.2
    epochs: int = 10
    mixed_precision: bool = False

class USSR1DCNNModel(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "USSR_1D_CNN_V1"

    def create_dataset(self):
        pass

    def init_model(self) -> bool:
        # Load validation dataset during initialization
        self.create_dataset()
        try:
            logger.info(f"Initializing UNet model")
            status = self.unet_model.init_model()
            logger.info(f"UNet model initialized")
            
            # Initialize each CNN model for each cluster
            logger.info(f"Initializing CNN models for clusters")
            for cluster_name, cnn1d_model in self.cnn1d_models.items():
                logger.info(f"Initializing CNN model for cluster: {cluster_name}")
                cnn1d_model.init_model()
                logger.info(f"CNN model for cluster {cluster_name} initialized")
                
            logger.info(f"USSR-1D-CNN model initialized")
            return status
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False
    
    def train(self, run_dir: str):
        logger.logger("No training done. Train 1DCNN and UNet models separately instead")
        history = {
            
        }
        return history, 0

    def validate_model(self):
        # Do reconstruction and see accuracy. 
        validate_reconstruction()

