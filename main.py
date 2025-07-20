from modules.model_runner.model_trainer import train_model
from modules.models.model_wrapper import ModelConfig
from modules.models.usrr_1dcnn.reconstruction.reconstruction import reconstruct_and_test
import logging
import argparse
from datetime import datetime
from modules.visualiser.visualiser import plot_upstream_conditions, visualise_rep_locations, plot_boundary_information, create_flood_animation, plot_extent_reference, plot_extent_prediction, plot_extents_on_same_image, visualise_area_check_map, plot_study_area, plot_study_area_clean
from modules.visualiser.metrics.performance_and_footprint import plot_metrics
from modules.visualiser.flow_analysis import find_peak_inflow_timestep, plot_hydrograph_clean
# from modules.visualiser.hydrological_visuals import find_peak_inflow_timestep
from modules.metrics_reader.metrics_reader import hyperparam_analysis
from modules.visualiser.test_event_viz import vizualise_test_event
from modules.models.usrr_1dcnn.reduction.rep_location_finder import find_representative_locations_and_clusters
from modules.datamanager.datamanager import create_inundation_map_tensors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

TRAIN_COMMAND = "train"
PREDICT_COMMAND = "predict"
GENERATE_SEQUENCES_COMMAND = "generate_sequences"
GENERATE_GRID_SEQUENCES_COMMAND = "generate_grid_sequences"
GENERATE_GRID_SEQUENCES_LIGHT_COMMAND = "generate_grid_sequences_light"
SRR_CLUSTER_COMMAND = "srr_cluster"
SRR_RECONSTRUCTION_COMMAND = "srr_reconstruction"
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
    return parser.parse_args()


def check_if_already_run(args):
    from modules.lib.constants import RUN_DIR
    import os
    import pandas as pd

    if args.tuning_mode:
        metrics_file = f"{RUN_DIR}/{args.model}/tuning_metrics.csv"
        hyperparams = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "input_time_len_h": args.input_time_len_h,
            "sampling_dist": args.sampling_dist,
            "n_clusters": args.n_clusters,
            "rl_group": args.rl_group,
            "lag": args.lag,
            "horizon": args.horizon,
            "patience": args.patience
        }
        compare_keys = ["batch_size", "learning_rate", "epochs", "patience", "lag", "horizon", "rl_group", "sampling_dist", "n_clusters", "input_time_len_h"]
    else:
        metrics_file = f"{RUN_DIR}/{args.model}/final_training_metrics.csv"
        # Define the hyperparameters to check for duplicates
        hyperparams = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "input_time_len_h": args.input_time_len_h,
            "sampling_dist": args.sampling_dist,
            "n_clusters": args.n_clusters,
            "rl_group": args.rl_group,
            "lag": args.lag,
            "horizon": args.horizon,
            "patience": args.patience,
            "dropout": args.dropout,
            "output_channel_size": args.output_channel_size,
            "fc_layer_size": args.fc_layer_size
        }

        # Only keep keys that are relevant for comparison
        compare_keys = ["batch_size", "learning_rate", "epochs", "patience", "lag", "horizon", "rl_group", "sampling_dist", "n_clusters", "input_time_len_h", "dropout", "output_channel_size", "fc_layer_size"]

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
                     SRR_RECONSTRUCTION_COMMAND, PLOT_COMMAND, METRICS_COMMAND, SRR_LSTM_REDUCTION_COMMAND]
    
    if args.command not in valid_commands:
        logger.error(f"Invalid command '{args.command}'")
        exit(1)

    elif args.command == TRAIN_COMMAND:
        if args.model == "USSR_1DCNN_V1":
            check_if_already_run(args)
            create_inundation_map_tensors()
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
        
    elif args.command == SRR_RECONSTRUCTION_COMMAND:
        if not args.sampling_dist or not args.n_clusters:
            logger.error("Missing required arguments for SRR reconstruction")
            exit(1)
        reconstruct_and_test(args.run_id, args.sampling_dist, args.n_clusters)
        
    elif args.command == SRR_LSTM_REDUCTION_COMMAND:
        from modules.models.srr_lstm.srr.srr_main import findRLS
        findRLS()
        
    elif args.command == PLOT_COMMAND:
        if not args.plot_type:
            logger.error("Plot type is required for plotting")
            exit(1)
            
        if args.plot_type == "up_conditions":
            if not args.event:
                logger.error("Event ID is required for upstream conditions plotting")
                exit(1)
            plot_upstream_conditions(args.event)
        elif args.plot_type == "rep_locations":
            if not args.run_id or not args.file:
                logger.error("Run ID and file are required for representative locations plotting")
                exit(1)
            visualise_rep_locations(args.run_id, args.file)
        elif args.plot_type == "boundary_information":
            plot_study_area()
        elif args.plot_type == "animation":
            create_flood_animation() 
        elif args.plot_type == "extent":
            plot_extents_on_same_image()
        elif args.plot_type == "1dcnn_extent":
            plot_extent_prediction()
        elif args.plot_type == "area_check":
            visualise_area_check_map()  
        elif args.plot_type == "plot_metrics":
            plot_metrics()
        elif args.plot_type == "architecture":
            # plot_model_architecture()
            pass
        elif args.plot_type == "study_area_clean":
            plot_study_area_clean()
        elif args.plot_type == "flow_analysis":
            find_peak_inflow_timestep()
            # find_peak_inflow_timestep()
        elif args.plot_type == "extent_reference":
            plot_extent_reference()
            
        elif args.plot_type == "test_event":
            vizualise_test_event()
            
        elif args.plot_type == "hydrograph_clean":
            plot_hydrograph_clean()
        else:
            logger.error(f"Unknown plot type: {args.plot_type}")
            exit(1)
            
    elif args.command == METRICS_COMMAND:
        hyperparam_analysis(args.model)
        
    logger.info("Commands executed successfully")

