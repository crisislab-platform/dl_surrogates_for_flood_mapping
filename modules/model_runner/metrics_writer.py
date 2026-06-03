from modules.model_runner.model_utils import get_model_parameters, count_total_neurons
import os
from modules.lib.constants import RUN_DIR
from modules.models.model_wrapper import Config
import logging
import pandas as pd
from modules.lib.constants import USRR_1DCNN_V1, USRR_UNET_V1, LSTM_SRR_V1, USRR_LSTM
from modules.metrics.plot_traning_metrics import plot_traning_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetricsWriter")

def save_training_metrics(run_id, history, train_time, model, model_config: Config, model_file, tuning_mode):   
    if history is None or history == {}:
        return 
    
    loss = history.get('loss', None)
    val_loss = history.get('val_loss', None)
    
    if tuning_mode:
        metrics_file = f'{RUN_DIR}/{model_config.model_name}/tuning_metrics.csv'
        metrics = {
            'run_id': run_id,
            'model': model_config.model_name,
            'loss': loss,
            'val_loss': val_loss, 
            'best_val_rmse': history.get('best_val_rmse', None),
            'best_epoch': history.get('best_epoch', None),
            'hyperparameters': history.get('hyperparameters', None),
            'train_time': float(train_time),
            'model_file': model_file
        }
        save_csv(metrics, metrics_file)
        plot_traning_metrics(history, os.path.join(RUN_DIR, model_config.model_name, run_id))
        return
    
    
    metrics_file = os.path.join(RUN_DIR, model_config.model_name, 'final_training_metrics.csv')
    memory_usage = str(history.get('memory', None))
    neurons = 0
    trainable_params = 0
    
    if model is not None:
        params = get_model_parameters(model)
        neurons = int(count_total_neurons(model))
        trainable_params = int(params['trainable_params'])
    else:
        trainable_params = history.get('trainable_params', 0)
        neurons = history.get('total_neurons', 0)

    metrics = {
        'run_id': run_id,
        'model': model_config.model_name,
        'total_neurons': neurons,
        'trainable_params':trainable_params,
        'loss': loss,
        'val_loss': val_loss, 
        'train_time': float(train_time),
        'model_file': model_file,
        'training_memory_usage': memory_usage, 
        'hyperparameters': history.get('hyperparameters', None),
    }
    save_csv(metrics, metrics_file)
    plot_traning_metrics(history, os.path.join(RUN_DIR, model_config.model_name, run_id))
    
     
def save_prediction_metrics(run_id, model_name, metrics):
    if model_name == USRR_1DCNN_V1 or model_name == USRR_UNET_V1 or model_name == LSTM_SRR_V1 or model_name == USRR_LSTM:
        metrics_file = os.path.join(RUN_DIR, model_name, 'final_performance_metrics.csv')
    else:
        metrics_file = f'{RUN_DIR}/final_performance_metrics.csv'
    try:
        metrics = {
            'run_id': run_id,
            'model': model_name,
            'mse': metrics.get('mse', ''),
            'rmse': metrics.get('rmse', ''),
            'inference_latency': metrics.get('pred_time', ''),
            'hit_rate': metrics.get('hit_rate', ''),
            'csi': metrics.get('csi', ''),
            'f2_score': metrics.get('f2_score', ''),
            'f3_score': metrics.get('f3_score', ''),
            'nse': metrics.get('nse', ''),
            'flops': metrics.get('flops', ''),
            'mRMSE': metrics.get('mRMSE', ''),
            'pred_memory_usage': metrics.get('pred_memory_usage', ''),
        }
        save_csv(metrics, metrics_file)
    except Exception as e:
        logger.error(f"Error updating prediction metrics: {e}")
        
def save_csv(metrics, metrics_file):
    import filelock
    
    # Create lock file path
    lock_file = f"{metrics_file}.lock"
    
    # Create a new DataFrame with the metrics
    new_row = pd.DataFrame([metrics])
    
    # Ensure lock file directory exists
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    # Use a file lock to prevent race conditions
    with filelock.FileLock(lock_file, timeout=60):
        file_exists = os.path.isfile(metrics_file)
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
            # Ensure directory exists
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            df.to_csv(metrics_file, index=False)
            logger.info(f"Metrics written to {metrics_file}")
        except Exception as e:
            logger.error(f"Error writing metrics to file: {e}")