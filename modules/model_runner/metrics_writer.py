from modules.model_runner.model_utils import get_model_parameters, count_total_neurons
import os
from modules.lib.constants import RUN_DIR
from modules.models.model_wrapper import ModelConfig
import logging
import pandas as pd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetricsWriter")

def save_metrics(run_id, history, train_time, model, model_config: ModelConfig, model_file):   
    metrics_file = f'{RUN_DIR}/training_metrics.csv' 
    file_exists = os.path.isfile(metrics_file)
    
    nurons = 0
    trainable_params = 0
    if model is not None:
        params = get_model_parameters(model)
        nurons = int(count_total_neurons(model))
        trainable_params = int(params['trainable_params'])
    
    loss, eval_loss, val_loss, eval_val_loss, memory_usage = None, None, None, None, None
    if history is not None or history != {}:
        loss = history.get('loss', None)
        eval_loss = history.get('eval_loss', None)
        val_loss = history.get('val_loss', None)
        eval_val_loss = history.get('eval_val_loss', None)
        memory_usage = str(history.get('memory', None))
    
    # Convert any NumPy types to native Python types to avoid JSON serialization issues
    metrics = {
        'run_id': run_id,
        'model': model_config.model_name,
        'total_neurons': nurons,
        'trainable_params':trainable_params,
        'flops': "",
        'loss': loss,
        'eval_loss': eval_loss,
        'val_loss': val_loss, 
        'eval_val_loss': eval_val_loss,
        'pred_mse': "",
        'pred_rmse': "",
        'pred_nse': "",
        'pred_rmse_wet': "",
        'pred_acc_wet': "",
        'pred_recall_wet': "",
        'pred_f1_wet': "",
        'pred_precision_wet': "",
        'train_time': float(train_time),
        'pred_time': "",
        'model_file': model_file,
        'training_memory_usage': memory_usage,
        'pred_memory_usage': "",
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
        
def save_prediction_metrics(run_id, metrics):
    metrics_file = f'{RUN_DIR}/training_metrics.csv'
    try:

        # Read existing DataFrame
        df = pd.read_csv(metrics_file)
        # Update the prediction metrics for the given run_id
        if run_id in df['run_id'].values:
            # Update metrics using dictionary values with safe access
            df.loc[df['run_id'] == run_id, 'pred_mse'] = metrics.get('mse', '')
            df.loc[df['run_id'] == run_id, 'pred_rmse'] = metrics.get('rmse', '')
            df.loc[df['run_id'] == run_id, 'pred_time'] = metrics.get('pred_time', '')
            df.loc[df['run_id'] == run_id, 'pred_nse'] = metrics.get('nse', '')
            df.loc[df['run_id'] == run_id, 'flops'] = metrics.get('flops', '')
            df.loc[df['run_id'] == run_id, 'pred_rmse_wet'] = metrics.get('wet_rmse', '')
            df.loc[df['run_id'] == run_id, 'pred_acc_wet'] = metrics.get('wet_acc', '')
            df.loc[df['run_id'] == run_id, 'pred_recall_wet'] = metrics.get('wet_recall', '')
            df.loc[df['run_id'] == run_id, 'pred_f1_wet'] = metrics.get('wet_f1', '')
            df.loc[df['run_id'] == run_id, 'pred_precision_wet'] = metrics.get('wet_precision', '')
            df.loc[df['run_id'] == run_id, 'pred_memory_usage'] = metrics.get('pred_memory_usage', '')
            
            # Write the updated DataFrame back to the file
            df.to_csv(metrics_file, index=False)
            logger.info(f"Prediction metrics updated for run ID {run_id}")
        else:
            logger.warning(f"Run ID {run_id} not found in metrics file")
    except Exception as e:
        logger.error(f"Error updating prediction metrics: {e}")