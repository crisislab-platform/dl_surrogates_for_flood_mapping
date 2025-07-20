from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.datamanager.raster.raster_loader_1dcnn import CNNRasterDataManager
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import POD_BNN_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PODBNN_ModelWrapper")
model_name  = POD_BNN_V1

class PODBNNModel(nn.Module):
    def __init__(self):
        super(PODBNNModel, self).__init__()
        pass
    
    def forward(self, x):
        pass
    
class PODBNNModelWrapper(ModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def init_model(self):
        return super().init_model()
    
    def create_dataset(self):
        pass
    
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        return super().test_model()
    