import gc
from modules.utils.path_util import OUTPUT_DIR
from modules.model_trainer.lstm.lstm import get_lstm_model
from lib.commons import ModelConfig
from datetime import datetime
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import pandas as pd
import logging
import time
import os

# Replace the logging setup with this
logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Make sure logger propagates
logger.propagate = True

def configure_gpu():
    """Configure GPU settings before any TF operations"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU configuration successful with {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.error(f"GPU configuration failed: {e}")

def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def count_total_neurons(model):
    """Count total number of neurons in the model"""
    total_neurons = 0
    for layer in model.layers:
        # Only count LSTM and Dense layers
        if isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.Dense)):
            # For LSTM layers, count the units
            if isinstance(layer, tf.keras.layers.LSTM):
                total_neurons += layer.units
            # For Dense layers, count the units (neurons)
            elif isinstance(layer, tf.keras.layers.Dense):
                total_neurons += layer.units
    return total_neurons

def get_model_parameters(model):
    """Get trainable and non-trainable parameters of the model"""
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': total_params
    }

def write_metrics(run_id, history, train_time, pred_time, model, model_config: ModelConfig):   
    metrics_file = f'{OUTPUT_DIR}/training_metrics.csv'    
    params = get_model_parameters(model)
    metrics = {
        'run_id': run_id,
        'model': model,
        'lag': model_config.lag,
        'horizon': model_config.horizon,
        'batch_size': model_config.batch_size,
        'learning_rate': model_config.learning_rate,
        'loss': str(history['loss']),
        'val_loss': str(history['val_loss']),
        'mae': str(history['mae']),
        'val_mae': str(history['val_mae']),
        'mse': str(history['mse']),
        'val_mse': str(history['val_mse']),
        'train_time': train_time,
        'pred_time': pred_time,
        'total_neurons': count_total_neurons(model),
        'trainable_params': params['trainable_params'],
        'non_trainable_params': params['non_trainable_params'],
        'total_params': params['total_params']
    }
    df = pd.DataFrame([metrics])
    df.to_csv(metrics_file, mode='a', index=False)

def train(model: tf.keras.Model, 
                train_dataset: tf.data.Dataset,
                val_dataset: tf.data.Dataset,
                training_steps: int = None,
                validation_steps: int = None,
                run_id: str = None,
                config: ModelConfig = None,
                epochs:int = 10,
                patience:int = 5) -> None:
    
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    try:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(run_dir, 'logs'),
                update_freq='epoch'
            )
        ]

        # Save model configuration
        with open(os.path.join(run_dir, 'model_config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)

        logger.info("Training model started")
        start_time = time.time()    
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=training_steps,
            validation_steps=validation_steps,
            verbose=1
        )
        
        end_time = time.time()
        train_time = end_time - start_time
        logger.info("Training completed in %.2f minutes" % ((end_time - start_time)/60))
        save_model_performane(history, run_id, run_dir, model, train_time, train_time, config)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        gc.collect()
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()

def save_model_performane(history, run_id, run_dir, model, train_time, pred_time, config: ModelConfig):
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
        plt.savefig(os.path.join(run_dir, 'model_history.png'))
        history_dict = history.history
        with open(os.path.join(run_dir, 'model_history.json'), 'w') as f:
            json.dump(history_dict, f)
        # Pass the model object to write_metrics
        write_metrics(run_id, history_dict, train_time, pred_time, model, config)
        
        # Save the model
        model.save(f"{run_dir}/{config.model_name}.h5")
        logger.info(f"Model saved in {run_dir}/{config.model_name}.h5")
    except Exception as e:
        logger.error(f"Error plotting model history: {e}")
        return
    
def train_model(lag: int, horizon: int, batch_size: int, learning_rate: float, 
                epochs: int = 10, patience: int = 2, model_name = None) -> str:
    
    run_id = generate_run_id()
    logger.info(f"Starting training run {run_id} with config: lag={lag}, horizon={horizon}, "
                f"batch_size={batch_size}, learning_rate={learning_rate}, "
                f"epochs={epochs}, patience={patience}")
    try:
        configure_gpu()
        if model_name == "LSTM_V1" or model_name == None:
            model, train_dataset, val_dataset, training_steps, validation_steps, model_config = get_lstm_model(lag, horizon, batch_size, learning_rate)
            train(model, train_dataset, val_dataset, training_steps, validation_steps, run_id, model_config, epochs, patience)
        elif model_name == "LSTM_V2":
            logger.error("Model not implemented")   
                    
        elif model_name == "1D_CNN_V1": 
            logger.error("Model not implemented")
        else:
            logger.error("Invalid model name")
        logger.info(f"Training run {run_id} completed")
        return run_id
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
    finally:
        gc.collect()
        tf.keras.backend.clear_session()



