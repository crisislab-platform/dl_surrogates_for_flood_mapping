from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import tensorflow as tf
from modules.dataloader.sequential.dataloader_optimised import FloodDataGenerator
from lib.constants import TRAIN_SUBSET_IDENTIFIER, VAL_SUBSET_IDENTIFIER
from lib.commons import ModelConfig
from modules.preprocessor.preprocessor import get_num_features
import logging 

logger = logging.getLogger("LSTM_Trainer")


def create_data_generators(lag, horizon, batch_size, epochs=5):
    # Create generators
    train_generator = FloodDataGenerator(
        batch_size=batch_size,
        lag=lag, 
        horizon=horizon,
        subset=TRAIN_SUBSET_IDENTIFIER, 
        epochs= epochs
    )
    
    val_generator = FloodDataGenerator(
        batch_size=batch_size,
        lag=lag,
        horizon=horizon,
        subset=VAL_SUBSET_IDENTIFIER, 
        epochs=epochs
    )
    
    train_dataset = train_generator.create_dataset()
    val_dataset = val_generator.create_dataset()
    
    train_steps = train_generator.calculate_steps_per_epoch()
    val_steps = val_generator.calculate_steps_per_epoch()
    
    return train_dataset, val_dataset, train_steps, val_steps

def create_lstm_model(config: ModelConfig) -> tf.keras.Model:
    strategy = tf.distribute.MirroredStrategy()
    num_features = get_num_features()
    with strategy.scope():
        try:
            model = Sequential([
                Input(shape=(config.lag, num_features), dtype='float32'),
                *[
                    Sequential([
                        LSTM(units, return_sequences=i < len(config.lstm_units)-1, 
                             dtype='float32',
                             kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
                        Dropout(config.dropout_rate)
                    ]) for i, units in enumerate(config.lstm_units)
                ],
                Dense(config.horizon, dtype='float32',
                      kernel_regularizer=tf.keras.regularizers.L2(1e-4))
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            logger.info(model.summary())
            return model
            
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            raise
        
        
def get_lstm_model(lag, horizon, batch_size, learning_rate) -> None:
    config = ModelConfig(
        lag=lag,
        horizon=horizon,
        batch_size=batch_size,
        learning_rate=learning_rate, 
        model_name="LSTM_V1"
    )
    model = create_lstm_model(config)
    train_dataset, val_dataset, training_steps, validation_steps = create_data_generators(config.lag, config.horizon, config.batch_size)
    return model, train_dataset, val_dataset, training_steps, validation_steps, config
        