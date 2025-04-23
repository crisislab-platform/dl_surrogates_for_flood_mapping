import logging
from dataclasses import dataclass
from modules.models.model_wrapper import ModelConfig, ModelWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GridLSTM_Trainer")

class LSTMSRRModelConfig(ModelConfig):
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    model_name: str
    dropout_rate: float = 0.2
    epochs: int = 10
    mixed_precision: bool = False

class LSTMSRRModel(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "LSTM_SRR_V1"
        
    def create_dataset(self):
        pass
    
    def init_model(self) -> bool:
        pass
    
    def validate_model(self, run_id = None, model_file = None):
        return "Not implemented"

        


