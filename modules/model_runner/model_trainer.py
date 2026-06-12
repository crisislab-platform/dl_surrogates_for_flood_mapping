from modules.models import model_wrapper
from modules.utils.run_util import generate_run_id
from modules.utils.path_util import ensure_dir
from modules.model_runner.model_factory import create_model
from modules.models.model_wrapper import Config
from modules.model_runner.metrics_writer import save_tuning_metrics, save_evaluation_metrics
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


def train_model(config: Config, args) -> str:
    run_id = generate_run_id()
    run_dir = os.path.join(RUN_DIR, config.model_name, run_id) 
    ensure_dir(run_dir)

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
        
        logger.info(f"Starting training run {run_id} with config: {model.config}")
        logger.info(f"Initializing model {config.model_name}")
        init_status = model.init_model()
        if not init_status:
            logger.error("Model initialization failed")
            return None
        
        logger.info(f"Training model {config.model_name}")
                
        tuning_mode = bool(args.tuning_mode)
        history, train_time, model_file = model.train(run_dir, tuning_mode)
        logger.info(f"Training completed in {train_time:.2f} seconds")
        logger.info(f"Training history: {history}")
        
        logger.info("Saving training history")
        if tuning_mode:
            save_tuning_metrics(run_id, history, train_time, model.model, config, model_file, tuning_mode)
            return run_id
         
        train_metrics = {
            'run_id': run_id,
            'model': config.model_name,
            'loss': history.get('loss', None),
            'train_time': float(train_time),
            'hyperparameters': history.get('hyperparameters', None),
        }
    

        logger.info("Testing model")
        metrics = model.test_model()
        
        save_evaluation_metrics(
            run_id, config.model_name, metrics, train_metrics,
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
