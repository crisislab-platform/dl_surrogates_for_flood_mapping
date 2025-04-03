from modules.model_runner.model_utils import get_model_parameters, count_total_neurons
import os
from modules.lib.constants import RUN_DIR
from modules.models.model_wrapper import ModelConfig
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetricsWriter")

def save_metrics(run_id, history, train_time, model, model_config: ModelConfig, model_file):   
    metrics_file = f'{RUN_DIR}/training_metrics.csv' 
    file_exists = os.path.isfile(metrics_file)
    params = get_model_parameters(model)
    
    # Convert any NumPy types to native Python types to avoid JSON serialization issues
    metrics = {
        'run_id': run_id,
        'model': model_config.model_name,
        'total_neurons': int(count_total_neurons(model)),
        'trainable_params': int(params['trainable_params']),
        'flops': "",
        'loss': str(history['loss'][-1]) if isinstance(history['loss'], list) else str(history['loss']),
        'eval_loss': str(history['eval_loss'][-1]) if isinstance(history['eval_loss'], list) else str(history['eval_loss']),
        'val_loss': str(history['val_loss'][-1]) if isinstance(history['val_loss'], list) else str(history['val_loss']), 
        'eval_val_loss': str(history['eval_val_loss'][-1]) if isinstance(history['eval_val_loss'], list) else str(history['eval_val_loss']),
        'pred_mse': "",
        'pred_rmse': "",
        'pred_nse': "",
        'pred_rmse_wet': "",
        'train_time': float(train_time),
        'pred_time': "",
        'model_file': model_file,
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
        
def save_prediction_metrics(run_id, pred_mse, pred_rmse, pred_nse,  pred_time, flops, rmse_wet):
    metrics_file = f'{RUN_DIR}/training_metrics.csv'
    try:
        # Read existing DataFrame
        df = pd.read_csv(metrics_file)
        
        # Update the prediction metrics for the given run_id
        if run_id in df['run_id'].values:
            df.loc[df['run_id'] == run_id, 'pred_mse'] = pred_mse
            df.loc[df['run_id'] == run_id, 'pred_rmse'] = pred_rmse
            df.loc[df['run_id'] == run_id, 'pred_time'] = pred_time
            df.loc[df['run_id'] == run_id, 'pred_nse'] = pred_nse
            df.loc[df['run_id'] == run_id, 'flops'] = flops
            df.loc[df['run_id'] == run_id, 'rmse_wet'] = rmse_wet
            # Write the updated DataFrame back to the file
            df.to_csv(metrics_file, index=False)
            logger.info(f"Prediction metrics updated for run ID {run_id}")
        else:
            logger.warning(f"Run ID {run_id} not found in metrics file")
    except Exception as e:
        logger.error(f"Error updating prediction metrics: {e}")