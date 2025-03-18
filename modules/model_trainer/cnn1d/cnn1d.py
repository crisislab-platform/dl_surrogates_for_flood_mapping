from modules.model_trainer.model import ModelConfig, Model
from modules.utils.path_util import RUN_DIR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from modules.dataloader.raster.raster_loader_1dcnn import create_raster_dataset
import tensorflow as tf
import numpy as np
import os
import logging
import time

logger = logging.getLogger("1DCNN")

class CNN1DModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "1DCNN_V1"
    
    def create_dataset(self):
        train_dataset, val_dataset, x_test, Y_test, steps, features, outputs = create_raster_dataset(
            self.config.batch_size, 
            self.config.lag, 
            self.config.horizon, 
            self.config.epochs
        ) 
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.x_test = x_test
        self.y_test = Y_test  # Note: Changed from Y_test to y_test for consistency
        self.steps = steps
        self.features = features
        self.outputs = outputs
        self.train_steps = len(train_dataset)
        self.val_steps = len(val_dataset)
        
    def init_model(self) -> bool:
        try:
            self.create_dataset()
            with tf.distribute.MirroredStrategy().scope():
                model = Sequential()
                model.add(Conv1D(32, kernel_size=1, activation = 'relu', input_shape=(self.steps, self.features)))
                model.add(Conv1D(128, activation = 'relu', kernel_size=1))
                model.add(Flatten())
                model.add(Dense(32, activation='relu'))
                model.add(Dense(256, activation='relu'))
                model.add(Dense(512, activation='relu'))
                model.add(Dense(self.outputs))
                optimizer = Adam(learning_rate=self.config.learning_rate)
                model.compile(loss='mse', metrics=['mse', 'mae'], optimizer=optimizer)
                self.model = model
                return True
        except Exception as e:
            logger.error(f"Error creating 1DCNN model: {e}")
            return False
