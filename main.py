from modules.model_trainer.model_trainer import train_model
from modules.db_initialiser.database_initialiser import init
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

TRAIN_COMMAND = "train"
DB_COMMAND = "db_init"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, default="train", help='Command to execute')
    parser.add_argument('--lag', type=int, default=8, help='Number of time steps to use as history')
    parser.add_argument('--horizon', type=int, default=1, help='Number of time steps to predict')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.command not in [TRAIN_COMMAND, DB_COMMAND]:
        logger.error(f"Invalid commands '{args.command}'")
        exit(1)
    
    logger.info(f"Executing command {args.command}")
    if args.command == TRAIN_COMMAND:
        train_model(
            lag=args.lag,
            horizon=args.horizon,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience
        )
    elif args.command == DB_COMMAND:
        init()
    logger.info("Commands executed successfully")
