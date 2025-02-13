# Create LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import logging
import time
from modules.dataloader.dataloader_optimised import FloodDataGenerator
import matplotlib.pyplot as plt
import json
from modules.preprocessor.preprocessor import get_num_features


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTM_model_training")

# Constants
LAG = 8 # 8 time steps of history. 2 hours  
FORECAST_HORIZON = 1 # Predict 1 time steps ahead. Next 15 minutes
TRAINING_BATCH_SIZE = 10000 # Number of samples per batch used for training
VALIDATION_BATCH_SIZE = 1000 # Number of samples per batch used for validation

TRAIN_SUBSET_IDENTIFIER = 'train'
VAL_SUBSET_IDENTIFIER = 'test'
PATIENCE = 2
EPOCHS = 10

def configure_gpu():
    """Configure GPU settings before any TF operations"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Enable mixed precision globally
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info(f"Mixed precision enabled with {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.error(f"GPU configuration failed: {e}")

def create_lstm_model(num_features):
    """Create LSTM model with GPU strategy"""
    strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
    with strategy.scope():
        model = Sequential([
            Input(shape=(LAG, num_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(FORECAST_HORIZON)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
    return model

def create_data_generators():
    train_generator = FloodDataGenerator(
        batch_size=TRAINING_BATCH_SIZE,
        lag=LAG, 
        horizon=FORECAST_HORIZON,
        subset=TRAIN_SUBSET_IDENTIFIER, 
        epochs=EPOCHS
    )
    
    val_generator = FloodDataGenerator(
        batch_size=VALIDATION_BATCH_SIZE,
        lag=LAG,
        horizon=FORECAST_HORIZON,
        subset=VAL_SUBSET_IDENTIFIER, 
        epochs=EPOCHS
    )
    
    # Create tf.data.Dataset objects
    train_dataset = train_generator.create_dataset()
    val_dataset = val_generator.create_dataset()
    train_steps = train_generator.calculate_steps()
    val_steps = val_generator.calculate_steps()
    return train_dataset, val_dataset, train_steps, val_steps

def train_lstm_model(model, train_dataset, val_dataset, training_steps, validation_steps):
    """Train model with GPU optimization"""
    try:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/{time.strftime("%Y%m%d-%H%M%S")}',
                update_freq='batch',
                profile_batch='100,120'
            )
        ]
        
        with tf.device('/GPU:0'):
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=EPOCHS,
                callbacks=callbacks,
                steps_per_epoch=training_steps,
                validation_steps=validation_steps,
                verbose=1
            )
        
        return model, history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def plot_model_training_history(history):
    # Plot the model history and save it as a figure and json file
    if not history:
        logger.error("Model history not found")
        return
    try: 
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
    except Exception as e:
        logger.error(f"Error plotting model history: {e}")
        return
        
if __name__ == "__main__":
    logger.info("Model trainer started")
    logger.info("TensorFlow version: {}".format(tf.__version__))
    
    # Configure GPU first
    configure_gpu()
    
    # Create model and datasets
    num_features = get_num_features()
    model = create_lstm_model(num_features)
    train_dataset, val_dataset, training_steps, validation_steps = create_data_generators()
    
    # Train model
    model, history = train_lstm_model(model, train_dataset, val_dataset, training_steps, validation_steps)
    plot_model_training_history(history)
    try:
        model.save("lstm_model.h5")
        logger.info("Model saved as lstm_model.h5")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    logger.info("Model training completed")



