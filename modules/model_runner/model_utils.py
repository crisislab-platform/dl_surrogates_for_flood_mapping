import torch.nn as nn
import os
import pandas as pd
import gc
from modules.lib.constants import RUN_DIR

def count_total_neurons(model):
    """Count total number of neurons in the model"""
    total_neurons = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_neurons

def get_model_parameters(model):
    """Get trainable and non-trainable parameters of the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': total_params
    }
    
def find_model_file(model_name):
    #Find the latest uent model file from the run directory
    run_dir = os.path.join(RUN_DIR, model_name)
    meta_data_file = os.path.join(run_dir, "final_training_metrics.csv")
    if not os.path.exists(meta_data_file):
        raise FileNotFoundError(f"Metadata file {meta_data_file} not found.")
    meta_data = pd.read_csv(meta_data_file)
    latest_run_id = meta_data['run_id'].sort_values(ascending=False).iloc[0]
    
    #find the model file for the latest run
    latest_model_file = meta_data.loc[meta_data['run_id'] == latest_run_id, 'model_file'].values[0]
    if not os.path.exists(latest_model_file):
        raise FileNotFoundError(f"Model file {latest_model_file} not found for run {latest_run_id}.")
    return latest_model_file