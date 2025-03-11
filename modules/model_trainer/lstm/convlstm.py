from tensorflow.keras.models import Model
from modules.model_trainer.model import Model as BaseModel
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Reshape, TimeDistributed, Conv2DTranspose
import tensorflow as tf
from lib.commons import ModelConfig
from modules.dataloader.sequential.sequence_loader import load_grid_datasets
from modules.dataloader.sequential.sequence_loader_light import load_grid_datasets_light
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GridConvLSTM_Trainer")

class ConvLSTMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "ConvLSTM_V1_LIGHT"
        
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
        with tf.distribute.MirroredStrategy().scope():
            try:
                # Enable mixed precision
                if self.config.mixed_precision:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    # Simple Input: (lag, 3 upstream features)
                    inputs = Input(shape=(self.config.lag, 3), name="input_layer")
                    
                    # Basic time-series processing layers
                    x = tf.keras.layers.Flatten()(inputs)  # Flatten to (batch, lag*3)
                    
                    # Simple dense layers
                    x = tf.keras.layers.Dense(256, activation='relu')(x)
                    x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
                    x = tf.keras.layers.Dense(512, activation='relu')(x)
                    x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
                    
                    # Output layer - directly predict flattened grid cells
                    output_size = self.grid_height * self.grid_width * self.config.horizon
                    outputs = tf.keras.layers.Dense(output_size)(x)
                    outputs = tf.keras.layers.Reshape((self.grid_height, self.grid_width, self.config.horizon))(outputs)
                    
                    model = Model(inputs=inputs, outputs=outputs)
                    
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                        loss='mse',
                        metrics=['mae', 'mse']
                    )
                    
                    model.summary(print_fn=logger.info)
                    self.model = model
                    return True
            except Exception as e:
                logger.error(f"Upstream-only model creation failed: {e}")
                raise
            
    def predict(self, run_id = None, model_file = None):
        return "Not implemented"