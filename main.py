from modules.model_trainer.model_trainer import train_model, predict_and_evaluate_only
from modules.model_trainer.model import ModelConfig
from modules.db_initialiser.database_initialiser import init
from modules.dataloader.sequential.file_sequence_generator import generate_grid_sequences_from_files, generate_grid_sequences_light_from_files
import logging
import argparse
import os
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

TRAIN_COMMAND = "train"
DB_COMMAND = "db_init"
PREDICT_COMMAND = "predict"
GENERATE_SEQUENCES_COMMAND = "generate_sequences"
GENERATE_GRID_SEQUENCES_COMMAND = "generate_grid_sequences"
GENERATE_GRID_SEQUENCES__LIGHT_COMMAND = "generate_grid_sequences_light"

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
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--window_length', type=int, default=None, help='Window length for data extraction')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    valid_commands = [TRAIN_COMMAND, DB_COMMAND, PREDICT_COMMAND, 
                     GENERATE_SEQUENCES_COMMAND, GENERATE_GRID_SEQUENCES_COMMAND, GENERATE_GRID_SEQUENCES__LIGHT_COMMAND]
    
    if args.command not in valid_commands:
        logger.error(f"Invalid command '{args.command}'")
        exit(1)
    
    logger.info(f"Executing command {args.command}")
    if args.command == GENERATE_GRID_SEQUENCES_COMMAND:
        generate_grid_sequences_from_files(args.lag, args.horizon)
    elif args.command == GENERATE_SEQUENCES_COMMAND:
        generate_grid_sequences_from_files(args.lag, args.horizon)
    elif args.command == GENERATE_GRID_SEQUENCES__LIGHT_COMMAND:
        generate_grid_sequences_from_files(args.lag, args.horizon, light=True)
    elif args.command == PREDICT_COMMAND:
        if not args.run_id:
            logger.error("Run ID is required for prediction")
            exit(1)
        predict_and_evaluate_only(args.run_id, args.model)
    elif args.command == TRAIN_COMMAND:
        # Use the unified ModelConfig
        config = ModelConfig(
            model_name=args.model,
            lag=args.lag,
            horizon=args.horizon,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            dropout_rate=args.dropout_rate,
            mixed_precision=args.mixed_precision
        )
        train_model(config)
    elif args.command == DB_COMMAND:
        init(window_length=args.window_length)
    logger.info("Commands executed successfully")
