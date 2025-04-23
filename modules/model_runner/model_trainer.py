from modules.utils.run_util import generate_run_id
from modules.utils.path_util import ensure_dir
from modules.model_runner.model_factory import create_model
from modules.models.model_wrapper import ModelConfig
from modules.visualiser.visualiser import plot_training_history
from modules.model_runner.metrics_writer import save_metrics, save_prediction_metrics
from modules.models.model_wrapper import ModelWrapper
from modules.lib.constants import RUN_DIR

import json
import torch
import logging
import os
import gc
import numpy as np

# Configure logger
logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)

def save_model_training_history(run_id, run_dir, model_wrapper:ModelWrapper, history, train_time, config, model_file):
    try:
        logger.info(f"history: {history}")
        logger.info("Writing training history to JSON")
        serializable_history = {}
        if history is None:
            logger.error("No training history available")
            return False
        for key, value in history.items():
            if isinstance(value, list):
                serializable_history[key] = [float(item) if isinstance(item, (np.number, np.ndarray)) 
                                            else item for item in value]
            elif isinstance(value, (np.number, np.ndarray)):
                serializable_history[key] = float(value)
            else:
                serializable_history[key] = value
                
        with open(os.path.join(run_dir, 'model_history.json'), 'w') as f:
            json.dump(serializable_history, f)
    
        logger.info("Plotting training history")
        plot_training_history(history, run_dir)
        
        logger.info("Saving training metrics")
        save_metrics(run_id, history, train_time, model_wrapper.model, config, model_file)
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def train_model(config: ModelConfig, args) -> str:
    run_id = generate_run_id()
    run_dir = os.path.join(RUN_DIR, config.model_name, run_id) 
    ensure_dir(run_dir)
    
    logger.info(f"Starting training run {run_id} with config: {vars(config)}")
    try:
        config.run_id = run_id
        config.run_dir = run_dir
        with open(os.path.join(run_dir, 'model_config.json'), 'w') as f:
            json.dump(vars(config), f)
        
        logger.info(f"Creating model instance {config.model_name}")
        model = create_model(config, args)
        if model is None:
            logger.error("Model creation failed")
            return None
        
        logger.info(f"Initializing model {config.model_name}")
        init_status = model.init_model()
        if not init_status:
            logger.error("Model initialization failed")
            return None
        logger.info(f"Training model {config.model_name}")
        
        
        history, train_time, model_file = model.train(run_dir)
        logger.info(f"Training completed in {train_time:.2f} seconds")
        logger.info("Training history: {history}")
        logger.info("Saving training history")
        
        state  = save_model_training_history(
            run_id, run_dir, model, history, train_time, config, model_file
        )
        if not state:
            logger.error("Model saving failed")
        else:
            logger.info("Model training history saved successfully")

        logger.info("Validating model")
        metrics = model.validate_model()
        
        # Extract metrics from the dictionary
        pred_mse = metrics.get("mse", 0)
        pred_rmse = metrics.get("rmse", 0)
        pred_nse = metrics.get("nse", 0)
        pred_time = metrics.get("pred_time", 0)
        flops = metrics.get("flops", 0)
        rmse_wet = metrics.get("wet_rmse", 0)
        wet_acc = metrics.get("wet_acc", 0)
    

        logger.info(f"Prediction completed in {pred_time:.2f} seconds")
        logger.info(f"Prediction MSE: {pred_mse}")
        logger.info(f"Prediction RMSE: {pred_rmse}")
        logger.info(f"Prediction NSE: {pred_nse}")
        logger.info(f"Model FLOPS: {flops}")
        logger.info(f"Wet cells RMSE: {rmse_wet}")
        logger.info(f"Wet cells classification accuracy: {wet_acc}")
        
        save_prediction_metrics(
            run_id, metrics
        )
        logger.info(f"Training run {run_id} completed")
        return run_id
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
