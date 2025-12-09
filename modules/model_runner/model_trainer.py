from modules.utils.run_util import generate_run_id
from modules.utils.path_util import ensure_dir
from modules.model_runner.model_factory import create_model
from modules.models.model_wrapper import ModelConfig
from modules.model_runner.metrics_writer import save_training_metrics, save_prediction_metrics
from modules.models.model_wrapper import ModelWrapper
from modules.lib.constants import RUN_DIR
from modules.lib.constants import USRR_CNN1D_COMBINED

import json
import torch
import logging
import os
import gc
import numpy as np

# Configure logger
logger = logging.getLogger("ModelTrainer")
logger.setLevel(logging.INFO)

def save_model_training_history(run_id, run_dir, model_wrapper:ModelWrapper, history, train_time, config, model_file, tuning_mode):
    try:
        logger.info("Saving training metrics")
        save_training_metrics(run_id, history, train_time, model_wrapper.model, config, model_file, tuning_mode)
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
                
        tuning_mode = bool(args.tuning_mode)
        history, train_time, model_file = model.train(run_dir, tuning_mode)
        logger.info(f"Training completed in {train_time:.2f} seconds")
        logger.info("Training history: {history}")
        
        logger.info("Saving training history")
        state  = save_model_training_history(
            run_id, run_dir, model, history, train_time, config, model_file, tuning_mode=tuning_mode
        )
        
        if not state:
            logger.error("Model saving failed")
        else:
            logger.info("Model training history saved successfully")

        if args.tuning_mode:
            logger.info("Tuning mode is enabled, skipping prediction")
            return run_id

        
        logger.info("Testing model")
        metrics = model.test_model()
        
        # Extract metrics from the dictionary
        pred_mse = metrics.get("mse", 0)
        pred_rmse = metrics.get("rmse", 0)
        pred_nse = metrics.get("nse", 0)
        pred_time = metrics.get("pred_time", 0)
        flops = metrics.get("flops", 0)
        mRMSE = metrics.get("mRMSE", 0)
        hit_rate = metrics.get("hit_rate", 0)
        csi = metrics.get("csi", 0)
        f2_score = metrics.get("f2_score", 0)
        f3_score = metrics.get("f3_score", 0)
    

        logger.info(f"Prediction completed in {pred_time:.2f} seconds")
        logger.info(f"Prediction MSE: {pred_mse}")
        logger.info(f"Prediction RMSE: {pred_rmse}")
        logger.info(f"Prediction mRMSE: {mRMSE}")
        logger.info(f"Prediction Hit Rate: {hit_rate}")
        logger.info(f"Prediction CSI: {csi}")
        logger.info(f"Prediction F2 Score: {f2_score}")
        logger.info(f"Prediction F3 Score: {f3_score}")
        logger.info(f"Prediction NSE: {pred_nse}")
        logger.info(f"Model FLOPS: {flops}")
        
        
        save_prediction_metrics(
            run_id, config.model_name, metrics
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
