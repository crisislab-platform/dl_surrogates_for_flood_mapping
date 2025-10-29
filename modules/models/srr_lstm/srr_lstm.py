from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.datamanager.point.sequential_loader_lstm import LSTMSequentialDataManager
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import LSTM_SRR_V1
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTMSRR_ModelWrapper")
model_name  = LSTM_SRR_V1

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)  # use the first dimension as batch_no
        self.hidden2 = nn.Linear(lstm_dim, output_dim)
    
    def forward(self, x):
        x = self.hidden1(x)
        _, (x, _) = self.lstm(torch.relu(x))
        x = self.hidden2(torch.tanh(x.view(-1, 1, self.lstm_dim)))
        return x
    
class LSTMModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        #Initialize model parameters
        self.input_dim = 3
        self.hidden_dim = config.args.get('hidden_size', 64)
        self.lstm_dim = config.args.get('fc_layer_size', 64)
        self.output_dim = None # To be set based on data
    
        #Arguments from config
        self.model_name = model_name
        self.device = check_device()
        self.lag = config.lag
        self.validation_event = self.config.fold  + 1
        self.input_time_len_h = config.args.get('input_time_len_h', False)
        self.rl_id = config.args.get('rl_id', None)
        self.tuninig_mode = config.args.get('tuning_mode', False)

    def init_model(self):
        self.create_dataset()
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            lstm_dim=self.lstm_dim,
            output_dim=self.output_dim
        ).to(self.device)
        self.loss_fn= nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return True
        
    def create_hyperparameters_dict(self):
        hyperparameters = {
            "learning_rate": self.config.learning_rate,
            'dropout': self.config.dropout,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "lag": self.config.lag,
            "horizon": self.config.horizon, 
            "rl_group": self.rl_id, 
            "input_time_len_h": self.input_time_len_h
        } 
        return hyperparameters
    
    def create_dataset(self):
        logger.info("Creating dataset")
        self.data_manager = LSTMSequentialDataManager(
            batch_size=self.config.batch_size,
            input_time_len_h=self.input_time_len_h,
            rl_id=self.rl_id,
            fold=self.validation_event,
            tuning_mode=self.tuninig_mode
        )
        
    def save_model_checkpoint(self, run_id, run_dir, model_state, config):
        try:
            model_file = os.path.join(run_dir, f"{config.model_name}_{run_id}.pth")
            torch.save({
                'model_state_dict': model_state,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.epochs,
                'run_id': run_id,
                'rl_id': self.rl_id,
                'input_time_len_h': self.input_time_len_h,
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'lstm_dim': self.lstm_dim,
                'output_dim': self.output_dim
            }, model_file) 
            return model_file      
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
            return None
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        return super().test_model()

    def save_predictions_at_points(self, predictions, ground_truth, points_csv):
        pass
    
    def save_predictions(self, pred):
        pass