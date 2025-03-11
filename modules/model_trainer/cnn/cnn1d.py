from lib.commons import ModelConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from modules.dataloader.raster.raster_loader import create_raster_dataset
import tensorflow as tf
import numpy as np
import pandas as pd
import os

import logging
logger = logging.getLogger("1DCNN")

def create_dataset(batch_size, lag, horizon, epochs=10): 
    return create_raster_dataset(batch_size, lag, horizon, epochs)

def create_1dcnn_model(steps, features, outputs, learning_rate):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        try:
            model = Sequential()
            model.add(Conv1D(32, kernel_size=1, activation = 'relu', input_shape=(steps, features)))
            model.add(Conv1D(128, activation = 'relu', kernel_size=1))
            model.add(Flatten())
            model.add(Dense(32, activation='relu'))
            model.add(Dense(256,activation='relu'))
            model.add(Dense(512,activation='relu'))
            model.add(Dense(outputs))
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(loss='mse', metrics=['mse', 'mae'], optimizer= optimizer)
            
        except Exception as e:
            logger.error(f"Error creating 1DCNN model: {e}")
            raise e
    return model

def get_1dcnn_model(batch_size, learning_rate, lag, horizon, epochs):
    config = ModelConfig(
        batch_size=batch_size,
        learning_rate=learning_rate, 
        model_name="1DCNN_V1", 
        lag=lag,
        horizon=horizon
    )
    train_dataset, val_dataset, x_test, Y_test , steps, features, outputs = create_dataset(batch_size, lag, horizon, epochs)
    trainig_steps = len(train_dataset)
    validation_steps = len(val_dataset)
    model  = create_1dcnn_model(steps, features, outputs, learning_rate)
    return model, train_dataset, val_dataset, x_test, Y_test , trainig_steps, validation_steps, config