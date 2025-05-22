from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.datamanager.raster.raster_loader_1dcnn import CNNRasterDataManager
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import CNN1D_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("1DCNN_ModelWrapper")
model_name  = CNN1D_V1

class CNNModel(nn.Module):
    def __init__(self, steps, features, outputs):
        super(CNNModel, self).__init__()
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
    
    def forward(self, x):
        # Conv1d expects input shape: [batch_size, channels, sequence_length]
        # But our data comes in as: [batch_size, sequence_length, features]
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

        # Second fully connected block with batch norm and dropout
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
     
        # Third fully connected block with batch norm
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = self.relu(x)
        
        # Output layer
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
            self.model = CNNModel(self.steps, self.features, self.outputs).to(self.device)
            self.loss_fn= nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            return True
        
        except Exception as e:
            logger.error(f"Error creating 1DCNN model: {e}")
            return False
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        return super().test_model()
    
    