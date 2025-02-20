# Create LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import logging
import time
from modules.model_trainer.archived.dataloader import FloodDataGenerator
import matplotlib.pyplot as plt
import json
from preprocessor import get_num_features


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTM_model_training")

logger.info("TensorFlow version: {}".format(tf.__version__))
logger.info("GPU Available: {}".format(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
logger.info(f"Number of GPUs available: {len(gpus)}")

# Constants
LAG = 8 # 8time steps of history. 2 hours  
FORECAST_HORIZON = 1 #Predict 1 time steps ahead. Next 15 minutes
TRAINING_BATCH_SIZE = 1000 # Number of samples per batch used for training

TRAIN_SUBSET_IDENTIFIER = 'train'
VAL_SUBSET_IDENTIFIER = 'test'
PATIENCE = 2
EPOCHS = 10

def create_lstm_model(num_features):
    input_shape = (LAG, num_features)  # (timesteps, features)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(FORECAST_HORIZON)  # Output shape matches horizon
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model
    
def create_data_generators(lag, horizon, batch_size):
    train_generator = FloodDataGenerator(batch_size,lag, horizon, TRAIN_SUBSET_IDENTIFIER)
    val_generator = FloodDataGenerator(batch_size, lag, horizon, VAL_SUBSET_IDENTIFIER)
    return train_generator, val_generator

# Train the model
def train_lstm_model(model, train_generator, val_generator):
    logger.info("Training LSTM model")
    model = create_lstm_model(num_features)
    start = time.time()
    logger.info("Training started at {}".format(start))
    
    call_backs = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=call_backs,
        verbose=1
    )

    end = time.time()
    logger.info(f"Training completed in {end-start} seconds")
    return model, history

def plot_model_training_history(history):
    # Plot the model history and save it as a figure and json file
    if not history:
        logger.error("Model history not found")
        return
    
    logger.info("Plotting model history")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('model_history.png')
    history_dict = history.history
    with open('model_history.json', 'w') as f:
        json.dump(history_dict, f)
        
        
if __name__ == "__main__":
    logger.info("Model trainer started")
    num_features = get_num_features()
    train_generator, val_generator= create_data_generators(LAG, FORECAST_HORIZON, TRAINING_BATCH_SIZE)
    model = create_lstm_model(num_features)
    model, history = train_lstm_model(model, train_generator, val_generator)
    plot_model_training_history(history)
    model.save("lstm_model.h5")
    logger.info("Model saved as lstm_model.h5")
    logger.info("Model training completed")
    
    # Evaluate the model by making predictions on the test set
    # evaluate_model(model, val_generator)
    
    
    


