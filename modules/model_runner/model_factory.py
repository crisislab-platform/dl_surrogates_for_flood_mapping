import logging
# from modules.models.lstm.lstm import SimpleLSTMModel
from modules.models.cnn1d.cnn1d import CNN1DSAModelWrapper
from modules.models.lstm_srr.lstm_srr import LSTMSRRModel
from modules.models.usrr_1dcnn.ussr1dcnn import USSR1DCNNModelWrapper
from modules.models.usrr_1dcnn.unet import UNetModelWrapper
from modules.models.usrr_1dcnn.cnn1d import CNN1DModelWrapper
from modules.models.model_wrapper import ModelConfig, ModelWrapper

from modules.lib.constants import (CNN1D_V1, LSTM_SRR_V1, 
                                  USRR_UNET_V1, USRR_1DCNN_V1, USRR_CNN1D_COMBINED)
logger = logging.getLogger("ModelFactory")

def create_model(config: ModelConfig, args)-> ModelWrapper:
    
    model_factories = {
        CNN1D_V1: lambda: create_1dcnn_standalone_model(config, args),
        
        USRR_UNET_V1: lambda: create_unet_model(config, args),
        
        USRR_1DCNN_V1: lambda: create_rl1dcnn_model(config, args),
        
        USRR_CNN1D_COMBINED: lambda: create_combined_model(config, args)
    }
    
    try:
        # Get the factory function for the requested model
        logger.info("Initializing factory")
        factory = model_factories.get(config.model_name)
        if factory:
            logger.info(f"Creating model: {config.model_name}")
            return factory()
        else:
            logger.error(f"Unknown model type: {config.model_name}")
            return None
    except Exception as e:
        logger.error(f"Error creating model {config.model_name}: {str(e)}")
        return None
    
def create_1dcnn_standalone_model(config: ModelConfig, args):
    cnn_model =  CNN1DSAModelWrapper(config)
    logger.info(f"Created 1DCNN model again")
    return cnn_model

def create_unet_model(config, args):
    config.args = {
        'sampling_dist': args.sampling_dist,
        'component_only': True
    }
    return UNetModelWrapper(config)

def create_rl1dcnn_model(config, args):
    sampling_dist = args.sampling_dist
    n_clusters = args.n_clusters
    rl_group = args.rl_group
    input_time_len_h = args.input_time_len_h
    if not sampling_dist or not n_clusters or not rl_group:
        logger.error("Missing required parameters for CNN1D model")
        raise ValueError("Missing required parameters for CNN1D model")
    
    config.args = { 
        'n_clusters': n_clusters,
        'sampling_dist': sampling_dist,
        'rl_group': rl_group,
        'input_time_len_h': input_time_len_h,
    }
    return CNN1DModelWrapper(config)

def create_combined_model(config, args):
    config.args = {
        'sampling_dist': args.sampling_dist,
        'n_clusters': args.n_clusters
    }
    
    # Create USSR1DCNNModel with combined configuration
    return USSR1DCNNModelWrapper(config)