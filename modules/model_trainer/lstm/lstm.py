from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Reshape, TimeDistributed, Flatten
import tensorflow as tf
from modules.model_trainer.model import Model, ModelConfig
from modules.dataloader.sequential.sequence_loader import load_grid_datasets
from modules.dataloader.sequential.sequence_loader_light import load_grid_datasets_light
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GridLSTM_Trainer")

# Model configuration
class LSTMModelConfig(ModelConfig):
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    model_name: str
    dropout_rate: float = 0.2
    epochs: int = 10
    mixed_precision: bool = False


class SimpleLSTMModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "GridLSTM_V1"
        
    def create_dataset(self):
        train_dataset, val_dataset, x_test, y_test, grid_height, grid_width, train_steps, val_steps = load_grid_datasets_light(
            self.config.batch_size, 
            self.config.epochs, 
            self.config.lag, 
            self.config.horizon
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.x_test = x_test
        self.y_test = y_test
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.train_steps = train_steps
        self.val_steps = val_steps
    
    def init_model(self) -> bool:
        self.create_dataset()
        num_cells = self.grid_height * self.grid_width
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            try:
                # Input shape: (lag, features)
                model = Sequential()
                model.add(LSTM(128, return_sequences=True, input_shape=(self.config.lag, 5)))
                model.add(LSTM(64))
                model.add(Dense(256, activation='relu'))
                model.add(Dense(512, activation='relu'))
                model.add(Dense(num_cells * self.config.horizon))
                model.add(Reshape((num_cells, self.config.horizon)))
            
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate= self.config.learning_rate),
                    loss='mse',
                    metrics=['mae', 'mse']
                )
                
                logger.info(f"Simple LSTM model created")
                logger.info(model.summary())
                self.model = model
                return True
            except Exception as e:
                logger.error(f"Grid LSTM model creation failed: {e}")
                raise
        
