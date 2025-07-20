import numpy as np 
import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import pandas as pd
from modules.preprocessor.preprocessor import Preprocessor, get_num_features, feature_columns  
from lib.constants import TRAIN_SUBSET_IDENTIFIER, VAL_SUBSET_IDENTIFIER, TEST_EVENT_ID
import logging
from datetime import datetime
import sys
import tensorflow as tf
from dotenv import load_dotenv
from modules.datamanager.sequential.sequence_generator import load_sequences

DB_BATCH_SIZE = 20000

logger = logging.getLogger("FloodDataGenerator")
logger.setLevel(logging.INFO)

# logger = setup_logger()
load_dotenv()

conn_params = {
    'dbname': 'carlisle_flood',
    'user': 'postgres',
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': 'localhost'
}

COUNT_QUERY = """SELECT COUNT(*) FROM flood_data_light"""

class FloodDataGenerator:
    
    def __init__(self, batch_size=None, lag=None, horizon=None, subset="train", epochs=5):
        self.batch_size = batch_size
        self.lag = lag
        self.horizon = horizon
        self.subset = subset
        self.epochs = epochs
        self.sequences_file = "flood_sequences.h5"
    
    def create_dataset(self) -> tf.data.Dataset:
        # Load pre-generated sequences
        X, y = load_sequences(self.sequences_file, self.subset)
        
        # Create dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Apply batching
        dataset = dataset.batch(self.batch_size)
        
        # Only repeat for training data
        if self.subset == TRAIN_SUBSET_IDENTIFIER:
            dataset = dataset.repeat(self.epochs)
        
        return dataset.prefetch(2)
    
    def calculate_steps_per_epoch(self):
        X, _ = load_sequences(self.sequences_file, self.subset)
        return max(1, len(X) // self.batch_size)
