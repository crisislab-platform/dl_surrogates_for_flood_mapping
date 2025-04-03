import torch.nn as nn
import os
import pandas as pd
import gc
from modules.lib.constants import RUN_DIR

def count_total_neurons(model):
    """Count total number of neurons in the model"""
    total_neurons = 0
    for module in model.modules():
        # Only count LSTM and Linear layers
        if isinstance(module, (nn.LSTM, nn.Linear)):
            # For LSTM layers, count the hidden_size
            if isinstance(module, nn.LSTM):
                total_neurons += module.hidden_size
            # For Linear layers, count the out_features (neurons)
            elif isinstance(module, nn.Linear):
                total_neurons += module.out_features
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
    
def find_model_file(model_name, run_id):
    #Find the latest uent model file from the run directory
    run_dir = os.path.join(RUN_DIR, model_name, run_id)
    model_file = os.path.join(run_dir, f"{model_name}_{run_id}.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found.")
    return model_file