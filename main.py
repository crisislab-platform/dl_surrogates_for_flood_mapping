import logging
import argparse
from datetime import datetime
import random

import numpy as np
import torch

from modules.model_runner.model_trainer import train_model
from modules.models.model_wrapper import Config
from modules.models.usrr_1dcnn.reduction.rep_location_finder import find_representative_locations_and_clusters
from utils.utils import check_if_already_run
from modules.metrics.metrics import metrics_analysis
import torch.multiprocessing as mp

TRAIN_COMMAND = "train"
SRR_CLUSTER_COMMAND = "srr_cluster"
METRICS_COMMAND = "metrics"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

valid_commands = [TRAIN_COMMAND,  SRR_CLUSTER_COMMAND,  METRICS_COMMAND]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, default="train", help='Command to execute')
    parser.add_argument('--model', type=str, default="1DCNN_V1", help='Model name for training or prediction')
    parser.add_argument('--run_id', type=str, help='Run ID for prediction')
    parser.add_argument('--lag', type=int, default=8, help='Number of time steps to use as history')
    parser.add_argument('--horizon', type=int, default=1, help='Number of time steps to predict')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--window_length', type=int, default=None, help='Window length for data extraction')
    parser.add_argument('--plot_type', type=str, help='Type of plot to generate')
    parser.add_argument('--file', type=str, help='Path to file for plotting')
    parser.add_argument('--event', type=str, help='Event ID for plotting')
    parser.add_argument('--rep_loc_file', type=str, help='Path to file containing representative locations')
    parser.add_argument('--sampling_dist', type=int, help='Sampling distance for representative locations')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters for SRR clustering')
    parser.add_argument('--random_state', type=int, help='Random state for SRR clustering')
    parser.add_argument('--n_init', type=int, help='Number of initializations for SRR clustering')
    parser.add_argument('--rl_group', type=str, help='RL group for clustering')
    parser.add_argument('--tuning_mode', action='store_true', help='Enable tuning mode')
    parser.add_argument('--fold', type=int, default=0, help='Validation fold for training')
    parser.add_argument('--input_time_len_h', type=float, default=False, help='Lenght of the input time series in hours')
    parser.add_argument('--physics_weight', type=float, default=0.5, help='Weight for physics-based loss')
    parser.add_argument('--usrr_conv_kernel', type=int, default=4, help='Directory to save run outputs')
    parser.add_argument('--usrr_pool_kernel', type=int, default=3, help='Pooling kernel size for USRR models')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model checkpoint')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model')
    parser.add_argument('--output_channel_size', type=int, default=1, help='Output channel size for the model')
    parser.add_argument('--fc_layer_size', type=int, default=64, help='Fully connected layer size for the model')
    parser.add_argument('--rl_id', type=str, default=None, help='ID for representative locations')
    parser.add_argument('--lstm_layers', type=int, default=5, help='Patience for early stopping during training')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for LSTM models')
    parser.add_argument('--train_events', type=int, nargs='+', default=[], help='Comma-separated list of training event IDs for the conditional diffusion model')
    parser.add_argument('--test_events', type=int, nargs='+', default=[], help='Comma-separated list of test event IDs for the conditional diffusion model')
    parser.add_argument('--validation_events', type=int, nargs='+', default=[], help='Comma-separated list of validation event IDs for the conditional diffusion model')
    parser.add_argument('--indices_per_timestep', type=int, default=1, help='Number of indices to consider per timestep')
    parser.add_argument('--study_area', type=str, default="carlisle", help='Study area for training or prediction')
    parser.add_argument('--tile_resolution', type=int, default=512, help='Tile resolution for spatial sampling')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--do_profile', action='store_true', help='Use profiling for performance analysis')
    parser.add_argument('--sigma', type=int, default=11, help='Sigma for Gaussian smoothing in representative location finding')
    parser.add_argument('--patch_domain', action='store_true', help='Whether to patch the spatial domain for training and inference')
    return parser.parse_args()


if __name__ == "__main__":
    

    args = parse_args()
    if args.command not in valid_commands:
        logger.error(f"Invalid command '{args.command}'")
        exit(1)
        
    #set random seed for variance
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    if args.command == TRAIN_COMMAND:
        mp.set_start_method('spawn', force=True)  
    
        config = Config(
            model_name=args.model,
            lag=args.lag,
            horizon=args.horizon,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience, 
            save_model=args.save_model,
            dropout=args.dropout,
            train_events=args.train_events,
            test_events=args.test_events,
            validation_events=args.validation_events,
            indices_per_timestep=args.indices_per_timestep,
            tuning_mode=args.tuning_mode,
            study_area=args.study_area, 
            random_seed=args.random_seed,
            do_profile=args.do_profile,
            patch_domain=args.patch_domain
        )
        
        logger.info(f"Training model with configuration: {config}")
        train_model(config, args)
        
    elif args.command == SRR_CLUSTER_COMMAND:
        if not args.sampling_dist or not args.n_clusters or not args.random_state or not args.n_init:
            logger.error("Missing required arguments for representative location clustering")
            exit(1)
            
        # Create a run_id for the run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        find_representative_locations_and_clusters(run_id, args.sampling_dist, args.n_clusters, args.random_state, args.n_init)
        
    elif args.command == METRICS_COMMAND:
        metrics_analysis()
    logger.info("Commands executed successfully")

