import gc
from modules.utils.path_util import OUTPUT_DIR
from modules.model_trainer.lstm.lstm import SimpleLSTMModel
from modules.model_trainer.lstm.convlstm import  ConvLSTMModel
from modules.model_trainer.cnn.cnn1d import CNN1DModel
from modules.model_trainer.model import ModelConfig
from datetime import datetime
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import pandas as pd
import logging
import time
import os
import numpy as np

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

def write_metrics(run_id, history, train_time, model, model_config: ModelConfig):   
    metrics_file = f'{OUTPUT_DIR}/training_metrics.csv'    
    params = get_model_parameters(model)
    metrics = {
        'run_id': run_id,
        'model':  model_config.model_name,
        'lag': model_config.lag,
        'horizon': model_config.horizon,
        'batch_size': model_config.batch_size,
        'learning_rate': model_config.learning_rate,
        'epochs': model_config.epochs,
        'patience': model_config.patience,
        'dropout_rate': model_config.dropout_rate,
        'mixed_precision': model_config.mixed_precision,
        'memory_limit': model_config.memory_limit,
        'loss': str(history['loss']),
        'val_loss': str(history['val_loss']),
        'mae': str(history['mae']),
        'val_mae': str(history['val_mae']),
        'mse': str(history['mse']),
        'val_mse': str(history['val_mse']),
        'train_time': train_time,
        'pred_time': "",
        'total_neurons': count_total_neurons(model),
        'trainable_params': params['trainable_params'],
        'non_trainable_params': params['non_trainable_params'],
        'total_params': params['total_params'], 
        'pred_mae': "",
        'pred_mse': "",
    }
    df = pd.DataFrame([metrics])
    df.to_csv(metrics_file, mode='a', index=False)
    
def save_model_and_performance(run_id, run_dir, model, history, train_time, config):
    """Save model, history, and metrics to disk"""
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
        write_metrics(run_id, history_dict, train_time, model, config)
        
        # Save the model with explicit metrics
        model_file_name = f"{run_dir}/{config.model_name}.h5"
        model.save(model_file_name, 
                  save_format='h5',
                  # Explicitly define loss and metrics to avoid serialization issues
                  options={
                      'save_traces': True,
                  })
        logger.info(f"Model saved in {model_file_name}")
        return model_file_name
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return

def predict_and_evaluate(run_id, run_dir,model_file_name, x_test, Y_test):
    """Evaluate model predictions on test data"""
    logger.info(f"Predicting and evaluating test data for run {run_id}")
    logger.info(f"Loading model from {model_file_name}")
    
    try:
        # Configure GPU
        configure_gpu()
        with tf.device('/GPU:0'):  # Explicitly use GPU
            # Load model with custom objects
            loaded_model = tf.keras.models.load_model(model_file_name, 
                custom_objects={
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.losses.MeanAbsoluteError()
                })
            
            logger.info(f"Test data shape: {x_test.shape}")
            logger.info(f"Ground truth shape: {Y_test.shape}")
            
            # Save x_test and Y_test to numpy files
            np.save(f'{run_dir}/x_test.npy', x_test)
            np.save(f'{run_dir}/Y_test.npy', Y_test)
            
            # Convert input data to tensors and move to GPU
            x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)
            
            predictions = []
            start_time = time.time()
            
            # Create batches for memory efficiency
            batch_size = 100
            num_samples = len(x_test)
            
            for i in range(0, num_samples, batch_size):
                batch_x = x_test[i:min(i + batch_size, num_samples)]
                pred = loaded_model(batch_x, training=False)  #Use model call directly instead of predict
                predictions.extend(pred.numpy())
                
                if (i + batch_size) % 1000 == 0:
                    logger.info(f"Processed {i + batch_size}/{num_samples} predictions")
            
            end_time = time.time()
            pred_time = end_time - start_time
            
            # Convert predictions list to numpy array
            predictions = np.array(predictions)
            
            # Calculate metrics on GPU using tf.keras.losses instead of metrics
            mse = tf.reduce_mean(tf.square(Y_test - predictions))
            mae = tf.reduce_mean(tf.abs(Y_test - predictions))
            rmse = tf.sqrt(mse)
            
            # Get values from tensors
            mse = float(mse.numpy())
            mae = float(mae.numpy())
            rmse = float(rmse.numpy())
            
            logger.info(f"Prediction completed in {pred_time:.2f} seconds")
            logger.info(f"Test MSE: {mse:.4f}")
            logger.info(f"Test MAE: {mae:.4f}")
            logger.info(f"Test RMSE: {rmse:.4f}")
            
            # Update metrics file
            metrics_file = f'{OUTPUT_DIR}/training_metrics.csv'
            df = pd.read_csv(metrics_file)
            df.loc[df['run_id'] == run_id, 'pred_time'] = pred_time
            df.loc[df['run_id'] == run_id, 'pred_mse'] = mse
            df.loc[df['run_id'] == run_id, 'pred_mae'] = mae
            df.loc[df['run_id'] == run_id, 'pred_rmse'] = rmse
            
            df.to_csv(metrics_file, index=False)
            logger.info(f"Prediction results saved in {metrics_file}")
            
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise
    finally:
        # Clear GPU memory
        tf.keras.backend.clear_session()
        gc.collect()

def predict_and_evaluate_only(run_id, model_name):
    """Predict and evaluate model using saved test data"""
    logger.info(f"Predicting and evaluating test data for run {run_id}")
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    model_file = os.path.join(run_dir, f"{model_name}.h5")
    
    try:
        # Load test data
        x_test = np.load(f'{run_dir}/x_test.npy')
        Y_test = np.load(f'{run_dir}/Y_test.npy')
        
        # Load model config
        with open(os.path.join(run_dir, 'model_config.json'), 'r') as f:
            config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        
        # Predict and evaluate
        predict_and_evaluate(model_file, x_test, Y_test, run_id, run_dir, config)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

def train_model(config: ModelConfig) -> str:
    run_id = generate_run_id()
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    logger.info(f"Starting training run {run_id} with config: {vars(config)}")
    
    try:
        configure_gpu()
        model = None
        if config.model_name == "LSTM_V1":
            model = SimpleLSTMModel(config)
            
        elif config.model_name == "1DCNN_V1": 
            model = CNN1DModel(config)
            
        elif config.model_name == "ConvLSTM_V1_LIGHT":
            model = ConvLSTMModel(config)  
        else:
            logger.error("Invalid model name")
            
        if model is not None:
            model.init_model()
            history, train_time = model.train()
            model_file_name = save_model_and_performance(run_id, run_dir, model, history, train_time, config)
            predict_and_evaluate(run_id, run_dir, model_file_name, model.x_test, model.y_test)
        logger.info(f"Training run {run_id} completed")
        return run_id
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
    finally:
        gc.collect()
        tf.keras.backend.clear_session()
