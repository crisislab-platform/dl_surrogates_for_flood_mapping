from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tcn import TCN
from modules.datamanager.raster.raster_loader_1dcnn import CNNRasterDataManager
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import TCN_V1
model_name  = TCN_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TCN_ModelWrapper")

class TCNModel(nn.Module):
    def __init__(self, steps, features, outputs):
        super(TCNModel, self).__init__()
        self.tcn = TCN(num_inputs=features,
                      num_channels=[32, 64],
                      dilations= [1,2],
                      kernel_size=3,
                      dropout=0.25,
                      causal=True, use_norm = 'weight_norm',
                      activation ='relu')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * steps, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, outputs)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.tcn(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Fully connected layers with batch norm and ReLU activation
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        return x
    
class TCNModelWrapper(ModelWrapper):
      
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        self.device = check_device()
        self.lag = config.lag
        self.steps = 1
        self.features = self.lag * 3
        self.outputs = 581061
        self.validation_event = self.config.fold  + 1
        self.tuninig_mode = config.args.get('tuning_mode', False)
    
    def create_dataset(self):
        logger.info("Creating dataset")
        self.data_manager = CNNRasterDataManager(self.config.lag, self.config.batch_size, validation_event = self.validation_event, tuning_mode=self.tuninig_mode)
        self.features = self.data_manager.features
        self.outputs = self.data_manager.outputs
        logger.info(f"Features: {self.features}, Outputs: {self.outputs}")
        
    def init_model(self) -> bool:
        try:
            self.create_dataset()
            self.model = TCNModel(self.steps, self.features, self.outputs).to(self.device)
            self.loss_fn= nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            return True
        
        except Exception as e:
            logger.error(f"Error creating TCN model: {e}")
            return False
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        return super().test_model()
    
    
    
    
    
    