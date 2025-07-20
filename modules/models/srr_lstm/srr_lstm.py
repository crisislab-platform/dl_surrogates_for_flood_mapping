from modules.models.model_wrapper import ModelConfig, ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from modules.datamanager.point.sequential_loader_lstm import LSTMSequentialDataManager
from modules.utils.run_util import check_device
import logging
from modules.lib.constants import LSTM_SRR_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTMSRR_ModelWrapper")
model_name  = LSTM_SRR_V1

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm_dim = lstm_dim
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
        self.input_dim = 15
        self.hidden_dim = 10
        self.lstm_dim = 20
        self.output_dim = 1
    
        #Arguments from config
        self.model_name = model_name
        self.device = check_device()
        self.lag = config.lag
        self.validation_event = self.config.fold  + 1
        self.tuninig_mode = config.args.get('tuning_mode', False)

    def init_model(self):
        self.create_dataset()
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            lstm_dim=self.lstm_dim,
            output_dim=self.output_dim
        )
        self.loss_fn= nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        return True
    
    def create_dataset(self):
        logger.info("Creating dataset")
        self.data_manager = LSTMSequentialDataManager(
            batch_size=self.config.batch_size,
            input_time_len_h=self.config.input_time_len_h,
            rl_group=self.config.rl_group,
            sampling_dist=self.config.sampling_dist,
            num_of_clusters=self.config.num_of_clusters,
            fold=self.validation_event,
            tuning_mode=self.tuninig_mode
        )
        
    def train(self, run_dir: str, tuning_mode = True) -> str:
        return super().train(run_dir, tuning_mode)
    
    def test_model(self):
        return super().test_model()
    