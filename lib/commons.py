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
    dropout_rate: float = 0.2
    epochs: int = 10
    mixed_precision: bool = False

