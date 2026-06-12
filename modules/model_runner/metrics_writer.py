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

def save_tuning_metrics(run_id, history, train_time, model, model_config: Config, model_file, tuning_mode):   
    if history is None or history == {}:
        return 
    
    metrics_file = f'{RUN_DIR}/{model_config.model_name}/tuning_metrics.csv'
    metrics = {
        'run_id': run_id,
        'model': model_config.model_name,
        'loss': history.get('loss', None),
        'val_loss': history.get('val_loss', None),
        'best_val_loss': history.get('best_val_loss', None),
        'best_epoch': history.get('best_epoch', None),
        'epochs_lr_reduced': history.get('epochs_lr_reduced', None),
        'hyperparameters': history.get('hyperparameters', None),
        'train_time': float(train_time),
        'model_file': model_file
    }
    save_csv(metrics, metrics_file)
    plot_traning_metrics(history, os.path.join(RUN_DIR, model_config.model_name, run_id))
    return

     
def save_evaluation_metrics(run_id, model_name, eval_metrics, train_metrics):
    metrics_file = os.path.join(RUN_DIR, model_name, 'final_metrics.csv')
    try:
        eval_metrics = {
            'run_id': run_id,
            'model': model_name,
            'mse': eval_metrics.get('mse', ''),
            'rmse': eval_metrics.get('rmse', ''),
            'mRMSE': eval_metrics.get('mRMSE', ''),
            'hit_rate': eval_metrics.get('hit_rate', ''),
            'csi': eval_metrics.get('csi', ''),
            'f2_score': eval_metrics.get('f2_score', ''),
            'flops': eval_metrics.get('flops', ''),
            'parameters': eval_metrics.get('parameters', ''),
            'peak_gpu_memory': eval_metrics.get('peak_gpu_memory', ''),
            'event_time': eval_metrics.get('event_time', ''),
            'timestep_time': eval_metrics.get('timestep_time', ''),
            'peak_volume_timing_error': eval_metrics.get('peak_volume_timing_error', ''),
            'true_peak_volume_timestep': eval_metrics.get('true_peak_volume_timestep', ''),
            'peak_volume_timestep': eval_metrics.get('peak_volume_timestep', ''),
            'train_history': train_metrics, 
        }
        save_csv(eval_metrics, metrics_file)
    except Exception as e:
        logger.error(f"Error updating evaluation metrics: {e}")
        
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