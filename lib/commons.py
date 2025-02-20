from dataclasses import dataclass
from typing import Tuple

# Model configuration
@dataclass
class ModelConfig:
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    model_name: str
    lstm_units: Tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.2

