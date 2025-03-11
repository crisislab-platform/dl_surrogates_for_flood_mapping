from tensorflow.keras.models import Sequential
from dataclasses import dataclass

# Model configuration
@dataclass
class ModelConfig:
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    model_name: str = "model"
    dropout_rate: float = 0.2
    epochs: int = 10
    mixed_precision: bool = False

class Model:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.train_dataset = None
        self.val_dataset = None
        self.x_test = None
        self.y_test = None
        self.grid_height = None
        self.grid_width = None
        self.train_steps = None
        self.val_steps = None
        self.model = None
        
    def init_model(self) -> bool:
        pass
    
    def create_dataset(self):
        pass