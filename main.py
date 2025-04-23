from modules.model_runner.model_trainer import train_model
from modules.models.model_wrapper import ModelConfig
from modules.models.usrr_1dcnn.spatial_reduction_module.rep_location_finder import find_representative_locations_and_clusters
from modules.models.usrr_1dcnn.spatial_reduction_module.reconstruction import validate_reconstruction
import logging
import argparse
from datetime import datetime
from modules.visualiser.visualiser import plot_upstream_conditions, visualise_rep_locations, plot_boundary_information, create_flood_animation, plot_extent_reference, plot_extent_prediction, plot_extents_on_same_image, visualise_area_check_map, draw_metrics
from modules.metrics_reader.metrics_reader import read_metrics


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
    parser.add_argument('--input_time_len_h', type=int, help='Input time length in hours for clustering')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    valid_commands = [TRAIN_COMMAND, PREDICT_COMMAND,  SRR_CLUSTER_COMMAND,
                     SRR_RECONSTRUCTION_COMMAND, PLOT_COMMAND, METRICS_COMMAND]
    
    if args.command not in valid_commands:
        logger.error(f"Invalid command '{args.command}'")
        exit(1)

    elif args.command == TRAIN_COMMAND:
        config = ModelConfig(
            model_name=args.model,
            lag=args.lag,
            horizon=args.horizon,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience
        )
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
        validate_reconstruction(args.run_id, args.sampling_dist, args.n_clusters)
        
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
            plot_boundary_information()
        elif args.plot_type == "animation":
            create_flood_animation() 
        elif args.plot_type == "extent":
            plot_extents_on_same_image()
        elif args.plot_type == "1dcnn_extent":
            plot_extent_prediction()
        elif args.plot_type == "area_check":
            visualise_area_check_map()  
        elif args.plot_type == "draw_metrics":
            draw_metrics()
        else:
            logger.error(f"Unknown plot type: {args.plot_type}")
            exit(1)
            
    elif args.command == METRICS_COMMAND:
        read_metrics()
        
    logger.info("Commands executed successfully")
