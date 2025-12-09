from modules.model_runner.model_trainer import train_model
from modules.models.model_wrapper import ModelConfig
import logging
import argparse
from datetime import datetime
from modules.metrics_reader.metrics_reader import hyperparam_analysis
from modules.models.usrr_1dcnn.reduction.rep_location_finder import find_representative_locations_and_clusters
from modules.datamanager.datamanager import create_inundation_map_tensors
from modules.visualiser.visualiser import plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

TRAIN_COMMAND = "train"
PREDICT_COMMAND = "predict"
GENERATE_SEQUENCES_COMMAND = "generate_sequences"
GENERATE_GRID_SEQUENCES_COMMAND = "generate_grid_sequences"
GENERATE_GRID_SEQUENCES_LIGHT_COMMAND = "generate_grid_sequences_light"
SRR_CLUSTER_COMMAND = "srr_cluster"
PLOT_COMMAND = "plot"
METRICS_COMMAND = "metrics"
SRR_LSTM_REDUCTION_COMMAND = "srr_lstm_reduction"

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
    parser.add_argument('--tuning_mode', type=bool, help='Enable tuning mode')
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
    return parser.parse_args()

def check_if_already_run(args):
    from modules.lib.constants import RUN_DIR
    import os
    import pandas as pd
   
    if args.model == "LSTM_SRR_V1":
        rl_group = args.rl_id
    else:
        rl_group = args.rl_group
        
    logger.info(f"Logger info {rl_group}")
    
    if args.tuning_mode:
        metrics_file = f"{RUN_DIR}/{args.model}/tuning_metrics.csv"
        hyperparams = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "input_time_len_h": args.input_time_len_h,
            "sampling_dist": args.sampling_dist,
            "n_clusters": args.n_clusters,
            "rl_group": rl_group,
            "lag": args.lag,
            "horizon": args.horizon,
            "patience": args.patience
        }
        compare_keys = ["batch_size", "learning_rate", "epochs", "patience", "lag", "horizon", "rl_group", "sampling_dist", "n_clusters", "input_time_len_h"]
    else:
        metrics_file = f"{RUN_DIR}/{args.model}/final_training_metrics.csv"
        # Define the hyperparameters to check for duplicates
        hyperparams = {
            # "batch_size": args.batch_size,
            # "learning_rate": args.learning_rate,
            # "epochs": args.epochs,
            # "input_time_len_h": args.input_time_len_h,
            # "sampling_dist": args.sampling_dist,
            # "n_clusters": args.n_clusters,
            "rl_group": rl_group,
            # "lag": args.lag,
            # "horizon": args.horizon,
            # "patience": args.patience,
            # "dropout": args.dropout,
            # "output_channel_size": args.output_channel_size,
            # "fc_layer_size": args.fc_layer_size
        }

        # Only keep keys that are relevant for comparison
        #compare_keys = ["batch_size", "learning_rate", "epochs", "patience", "lag", "horizon", "rl_group", "sampling_dist", "n_clusters", "input_time_len_h", "dropout", "output_channel_size", "fc_layer_size"]
        compare_keys = ["rl_group"]
    if not os.path.exists(metrics_file):
        return  # No previous runs, so continue

    try:
        df = pd.read_csv(metrics_file)
    except Exception as e:
        logger.warning(f"Could not read {metrics_file}: {e}")
        return
    
    # Prepare current run's hyperparameters as strings for comparison
    current_hyperparams = {k: hyperparams[k] for k in compare_keys if k in hyperparams}

    import ast
    for idx, row in df.iterrows():
        if row.get("model") != args.model:
            continue
        if args.tuning_mode and row.get("fold") != args.fold:
            continue
        try:
            row_hyperparams = ast.literal_eval(row.get("hyperparameters", "{}"))
        except Exception:
            continue
        # Only compare keys present in both
        if all(str(row_hyperparams.get(k)) == str(current_hyperparams.get(k)) for k in current_hyperparams):
            logger.warning("A run with the same hyperparameters already exists. Exiting to avoid duplicate runs.")
            exit(0)


if __name__ == "__main__":
    args = parse_args()
    valid_commands = [TRAIN_COMMAND, PREDICT_COMMAND,  SRR_CLUSTER_COMMAND,
                     PLOT_COMMAND, METRICS_COMMAND, SRR_LSTM_REDUCTION_COMMAND]
    
    if args.command not in valid_commands:
        logger.error(f"Invalid command '{args.command}'")
        exit(1)

    elif args.command == TRAIN_COMMAND:
            
        if args.model == "USSR_1DCNN_V1" or args.model == "USRR_LSTM_V1":
            # check_if_already_run(args)
            create_inundation_map_tensors()
            
        if args.model == "LSTM_SRR_V1":
            check_if_already_run(args)
            
        config = ModelConfig(
            model_name=args.model,
            lag=args.lag,
            horizon=args.horizon,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience, 
            fold=args.fold,
            save_model=args.save_model,
            dropout=args.dropout,
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
        
    elif args.command == SRR_LSTM_REDUCTION_COMMAND:
        from modules.models.srr_lstm.srr.srr_main import findRLS
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        findRLS(run_id)
        
    elif args.command == PLOT_COMMAND:
        plot(args.plot_type)
            
    elif args.command == METRICS_COMMAND:
        hyperparam_analysis(args.model)
        
    logger.info("Commands executed successfully")

