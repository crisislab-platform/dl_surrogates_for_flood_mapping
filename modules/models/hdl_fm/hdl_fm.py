from modules.models.model_wrapper import Config, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import HDL_FM_V1
from modules.datamanager.hdlfm.hdl_fm_dm import HDLFMDataManager
from torch.profiler import profile, ProfilerActivity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HDLFM_ModelWrapper")
model_name  = HDL_FM_V1

class HDLFMModel(nn.Module):
    
    def __init__(self, output_shape):
        super(HDLFMModel, self).__init__()
        
        # CNN layers
        input_channels = 3
        cnn1_channels = 13
        cnn2_channels = 5
        H = output_shape[0]
        W = output_shape[1]

        # Calculate dimensions after CNN layers with pooling
        H_after_cnn = H // (4*3*2)  # Three max-pooling layers with stride 2
        W_after_cnn = W // (4*3*2)  # Three max-pooling layers with stride 2
        lstm_input_size = H_after_cnn * W_after_cnn  # Size after flattening
        lstm_hidden_size = lstm_input_size
        lstm_num_layers = 1
    
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn1_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(cnn1_channels, cnn2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(cnn2_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #Flatten Layer
        self.flatten = nn.Flatten()
        
        #LSTM Layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        #Linear Layer
        self.linear = nn.Linear(lstm_hidden_size, H*W)
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.size(0)
        
        # CNN
        x = self.cnn(x)
        
        # Flatten
        x = self.flatten(x)
    
        #LSTM
        x, _ = self.lstm(x.unsqueeze(1))  # Add sequence dimension

        #Linear
        x = self.linear(x)
        return x
    
class HDLFMModel_Westport(nn.Module):
    def __init__(self, output_shape):
        super(HDLFMModel_Westport, self).__init__()
        
        H, W = output_shape  # 2448, 2416
        
        input_channels = 3
        cnn1_channels = 13
        cnn2_channels = 5
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn1_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn1_channels, cnn2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn2_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((50, 50))
        
        lstm_input_size = 2500
        # lstm_hidden_size = lstm_input_size
        # lstm_num_layers = 1
        
        # # Flatten layer
        self.flatten = nn.Flatten()
        
        # # LSTM layer
        # self.lstm = nn.LSTM(input_size=lstm_input_size, 
        #                     hidden_size=lstm_hidden_size, 
        #                     num_layers=lstm_num_layers, 
        #                     batch_first=True)
        
        # Linear layer
        self.linear = nn.Linear(lstm_input_size, H*W)

    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        
        # Flatten
        x = self.flatten(x)
    
        #LSTM
        # x, _ = self.lstm(x.unsqueeze(1))  # Add sequence dimension

        #Linear
        x = self.linear(x)
        return x
                            
        
    
class HDLFMModelWrapper(ModelWrapper):
    def __init__(self, config: Config):
        super().__init__(config)
        self.model_name = model_name
        self.device = check_device()

    def create_dataset(self):
        logger.info("Creating HDLFM dataset")
        self.data_manager = HDLFMDataManager(self.config)
        self.input_features = self.data_manager.input_features
        self.outputs = self.data_manager.output_shape
        logger.info(f"Features: {self.input_features}, Outputs: {self.outputs}")
    
    def init_model(self):
        self.create_dataset()
        if self.config.study_area == "westport":
            self.model = HDLFMModel_Westport(self.outputs).to(self.device).float()
        else:
            self.model = HDLFMModel(self.outputs).to(self.device).float()
        self.loss_fn= nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=20,
        )
        return True
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        
    