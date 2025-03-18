import gc
from modules.utils.path_util import RUN_DIR
from modules.utils.run_util import generate_run_id
from modules.model_trainer.lstm.lstm import SimpleLSTMModel
# from modules.model_trainer.lstm.convlstm import  ConvLSTMModel
from modules.model_trainer.cnn1d.cnn1d import CNN1DModel
from modules.model_trainer.model import ModelConfig
from modules.model_trainer.lstm_srr.lstm_srr import LSTMSRRModel
from modules.model_trainer.ussr_1dcnn.ussr1dcnn import USSR1DCNNModel
from datetime import datetime
from modules.utils.path_util import ensure_dir
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
    """Write training metrics to a CSV file"""
    metrics_file = f'{RUN_DIR}/training_metrics.csv' 
    
    # Check if the file exists
    file_exists = os.path.isfile(metrics_file)
    
    params = get_model_parameters(model)
    
    # Convert any NumPy types to native Python types to avoid JSON serialization issues
    metrics = {
        'run_id': run_id,
        'model': model_config.model_name,
        'lag': int(model_config.lag),
        'horizon': int(model_config.horizon),
        'batch_size': int(model_config.batch_size),
        'learning_rate': float(model_config.learning_rate),
        'epochs': int(model_config.epochs),
        'patience': int(model_config.patience),
        'dropout_rate': float(model_config.dropout_rate),
        'mixed_precision': bool(model_config.mixed_precision),
        'memory_limit': None if model_config.memory_limit is None else int(model_config.memory_limit),
        'loss': str(history['loss'][-1]) if isinstance(history['loss'], list) else str(history['loss']),
        'val_loss': str(history['val_loss'][-1]) if isinstance(history['val_loss'], list) else str(history['val_loss']),
        'mae': str(history['mae'][-1]) if isinstance(history['mae'], list) else str(history['mae']),
        'val_mae': str(history['val_mae'][-1]) if isinstance(history['val_mae'], list) else str(history['val_mae']),
        'mse': str(history['mse'][-1]) if isinstance(history['mse'], list) else str(history['mse']),
        'val_mse': str(history['val_mse'][-1]) if isinstance(history['val_mse'], list) else str(history['val_mse']),
        'train_time': float(train_time),
        'pred_time': "",
        'total_neurons': int(count_total_neurons(model)),
        'trainable_params': int(params['trainable_params']),
        'non_trainable_params': int(params['non_trainable_params']),
        'total_params': int(params['total_params']), 
        'pred_mae': "",
        'pred_mse': "",
        'pred_rmse': "",
    }
    
    # Create a new DataFrame with the metrics
    new_row = pd.DataFrame([metrics])
    
    if file_exists:
        try:
            # Read existing DataFrame
            df = pd.read_csv(metrics_file)
            # Append new row to existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading existing metrics file: {e}")
            # If there's an error reading the file, create a new DataFrame
            df = new_row
    else:
        # If file doesn't exist, use the new DataFrame
        df = new_row
    
    # Write the combined DataFrame back to the file
    try:
        df.to_csv(metrics_file, index=False)
        logger.info(f"Metrics written to {metrics_file}")
    except Exception as e:
        logger.error(f"Error writing metrics to file: {e}")
        
def plot_training_history(history, run_dir):
    """Save model, history, and metrics to disk"""
    if not history:
        logger.error("Model history not found")
        return None

    logger.info("Plotting model history")
    plt.figure()  # Create a new figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(run_dir, 'model_history.png'))
    plt.close()  # Close the figure to free memory

def save_model_and_performance(run_id, run_dir, model, history, train_time, config):
    try:
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(x) for x in value]
        plot_training_history(history, run_dir)
        
        with open(os.path.join(run_dir, 'model_history.json'), 'w') as f:
            json.dump(history_dict, f)
            
        # Pass the model object to write_metrics
        write_metrics(run_id, history_dict, train_time, model, config)
        
        # Save the model with explicit metrics
        model_file_name = f"{run_dir}/{config.model_name}.h5"
        model.save(model_file_name)
        logger.info(f"Model saved in {model_file_name}")
        return model_file_name
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def save_model_performance_torch(run_id, run_dir, history, train_time, model, config: ModelConfig):
    plot_training_history(history, run_dir)
    with open(os.path.join(run_dir, 'model_history.json'), 'w') as f:
            json.dump(history, f)
        
    metrics_file = f'{RUN_DIR}/training_metrics_unet.csv' 
    
    # Check if the file exists
    file_exists = os.path.isfile(metrics_file)
    metrics = {
        'run_id': run_id,
        'model': config.model_name,
        'batch_size': int(config.batch_size),
        'learning_rate': float(config.learning_rate),
        'epochs': int(config.epochs),
        'losses': str(history['losses'][-1]) if isinstance(history['losses'], list) else str(history['loss']),
        'eval_losses': str(history['val_losses'][-1]) if isinstance(history['val_losses'], list) else str(history['eval_losses']),
        'val_losses': str(history['val_losses'][-1]) if isinstance(history['val_losses'], list) else str(history['val_losses']),
        'train_time': float(train_time)
    }
    
    # Create a new DataFrame with the metrics
    new_row = pd.DataFrame([metrics])
    if file_exists:
        try:
            # Read existing DataFrame
            df = pd.read_csv(metrics_file)
            # Append new row to existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading existing metrics file: {e}")
            df = new_row
    else:
        df = new_row
    try:
        df.to_csv(metrics_file, index=False)
        logger.info(f"Metrics written to {metrics_file}")
    except Exception as e:
        logger.error(f"Error writing metrics to file: {e}")
        
    # Save the model with explicit metrics
    model_file_name = f"{run_dir}/{config.model_name}.h5"
    model.save(model_file_name)
    logger.info(f"Model saved in {model_file_name}")


def predict_and_evaluate(run_id, run_dir, model_file_name, x_test, y_test):
    """Evaluate model predictions on test data"""
    logger.info(f"Predicting and evaluating test data for run {run_id}")
    
    # Check if model_file_name is valid
    if not model_file_name or not os.path.exists(model_file_name):
        logger.error(f"Model file not found: {model_file_name}")
        return
        
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
            logger.info(f"Ground truth shape: {y_test.shape}")
            
            # Save x_test and y_test to numpy files
            np.save(f'{run_dir}/x_test.npy', x_test)
            np.save(f'{run_dir}/y_test.npy', y_test)
            
            # Convert input data to tensors and move to GPU
            x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)
            
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
            mse = tf.reduce_mean(tf.square(y_test_tensor - predictions))
            mae = tf.reduce_mean(tf.abs(y_test_tensor - predictions))
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
            metrics_file = f'{RUN_DIR}/training_metrics.csv'
            try:
                df = pd.read_csv(metrics_file)
                # Find the row with the matching run_id and update it
                if run_id in df['run_id'].values:
                    df.loc[df['run_id'] == run_id, 'pred_time'] = pred_time
                    df.loc[df['run_id'] == run_id, 'pred_mse'] = mse
                    df.loc[df['run_id'] == run_id, 'pred_mae'] = mae
                    df.loc[df['run_id'] == run_id, 'pred_rmse'] = rmse
                    df.to_csv(metrics_file, index=False)
                    logger.info(f"Prediction results updated in {metrics_file}")
                else:
                    logger.warning(f"Run ID {run_id} not found in metrics file")
            except Exception as e:
                logger.error(f"Error updating metrics file: {e}")
            
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
    run_dir = os.path.join(RUN_DIR, run_id)
    model_file = os.path.join(run_dir, f"{model_name}.h5")
    
    # Check if the model file exists
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return
        
    try:
        # Load test data
        x_test = np.load(f'{run_dir}/x_test.npy')
        y_test = np.load(f'{run_dir}/y_test.npy')  # Use y_test instead of Y_test
        
        # Load model config
        with open(os.path.join(run_dir, 'model_config.json'), 'r') as f:
            config_dict = json.load(f)
            config = ModelConfig(**config_dict)
            
        # Predict and evaluate with the right parameter order
        predict_and_evaluate(run_id, run_dir, model_file, x_test, y_test)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

def train_model(config: ModelConfig, args) -> str:
    run_id = generate_run_id()
    run_dir = os.path.join(RUN_DIR, run_id)
    ensure_dir(run_dir)
    logger.info(f"Starting training run {run_id} with config: {vars(config)}")
    config.run_id = run_id
    config.run_dir = run_dir
    try:
        configure_gpu()
        model = None
        if config.model_name == "LSTM_V1":
            model = SimpleLSTMModel(config)
            
        elif config.model_name == "1DCNN_V1": 
            model = CNN1DModel(config)
            
        elif config.model_name == "LSTM_SRR_V1":
            model = LSTMSRRModel(config)
            
        elif config.model_name == "USSR_1D_CNN_V1":
            config.args = {
                'rep_loc_file_path': args.rep_loc_file, 
                'sampling_dist': args.sampling_dist
            }
            model = USSR1DCNNModel(config)
            
        else:
            logger.error("Invalid model name")
            
        if model is not None:
            status = model.init_model()
            if not status:
                logger.error("Model initialization failed")
                return None
            history, train_time = model.train(run_dir)
            # Check if save_model_and_performance returned a valid file name
            model_file_name = save_model_and_performance(run_id, run_dir, model.model, history, train_time, config)
            if model_file_name:  # Only call predict if we have a valid model file
                predict_and_evaluate(run_id, run_dir, model_file_name, model.x_test, model.y_test)
            else:
                logger.error("Model file name is None, skipping prediction")      
        logger.info(f"Training run {run_id} completed")
        return run_id
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
    finally:
        gc.collect()
        tf.keras.backend.clear_session()
