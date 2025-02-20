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
    
    def __init__(self, batch_size = None, lag = None, horizon = None, subset = "train", is_preprocessor = False, epochs = 5):
        self.batch_size = batch_size
        self.lag = lag
        self.horizon = horizon
        self.subset = subset
        self.test_subset_event = TEST_EVENT_ID
        self.is_preprocessor = is_preprocessor
        self.pool = SimpleConnectionPool(1, 10, **conn_params)
        self.epochs = epochs
        self.device = None  # Remove device specification
        # if is_preprocessor:
        #    return
        # self.preprocessor = Preprocessor() # Skip preprocessing for now
    
    def generate_sequences(self):
        try:
            with self.pool.getconn() as conn:
                with conn.cursor('flood_cursor', scrollable=True) as cursor:
                    cursor.itersize = DB_BATCH_SIZE
                    query = f"""
                        SELECT
                            event_id,
                            cell_id,
                            timestep,
                            upstream1,
                            upstream2,
                            upstream3,
                            elevation,
                            depth
                        FROM flood_data_light
                        WHERE event_id {'=' if self.subset == 'test' else '!='} %s
                        ORDER BY event_id, cell_id, timestep
                    """
                    cursor.execute(query, (self.test_subset_event,))
            
                    # create sequences based on event, cell_id, and timestep
                    buffer = []
                    current_cell_id = None
                    current_event = None
                    for record in cursor:
                        if current_event is None:
                            current_event = record[0]
                        if current_cell_id is None:
                            current_cell_id = record[1]
                        if current_event != record[0] or current_cell_id != record[1]:
                            if len(buffer) >= self.lag + self.horizon:
                                df = pd.DataFrame(buffer, columns=['event_id', 'cell_id', 'timestep'] + feature_columns)
                                # features = self.preprocessor.preprocess(df, self.subset == TRAIN_SUBSET_IDENTIFIER)[feature_columns]
                                features = df [feature_columns] # Skip preprocessing for now
                                while len(buffer) >= self.lag + self.horizon:
                                    x = features[:self.lag].values
                                    y = features[self.lag:self.lag + self.horizon]['depth'].values
                                    buffer.pop(0)
                                    yield x, y
                            buffer = []
                            current_cell_id = record[1]
                            current_event = record[0]
                        buffer.append(record)               
        finally:
            if hasattr(self, 'pool'):
                self.pool.putconn(conn)

    def create_dataset(self) -> tf.data.Dataset:
        # Create dataset without device specification
        dataset = tf.data.Dataset.from_generator(
            self.generate_sequences,
            output_signature=(
                tf.TensorSpec(shape=(self.lag, get_num_features()), dtype=tf.float32),
                tf.TensorSpec(shape=(self.horizon,), dtype=tf.float32)
            )
        )
        if self.subset == TRAIN_SUBSET_IDENTIFIER:
            return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE).repeat(self.epochs)
        else:
            return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE).repeat(self.epochs)   
    
    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.closeall()
            
    def calculate_steps_per_epoch(self):
        # Calculate total number of sequences
        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    if self.subset == 'test':
                        op = '='
                    else:
                        op = '!='
                    
                    cur.execute(f"""
                        SELECT COUNT(DISTINCT cell_id) FROM flood_data_light
                        WHERE event_id {op} {self.test_subset_event}
                    """)
                    cell_id_count =  cur.fetchone()[0] or 0
                    logger.info(f"Cell ID count: {cell_id_count}")
                    
                    # For each event_id get the timestep count
                    cur.execute(f"""
                        SELECT event_id,COUNT(DISTINCT timestep) FROM flood_data_light
                        WHERE event_id {op} {self.test_subset_event}
                        GROUP BY event_id
                    """)
                    timestep_counts = cur.fetchall()
                    total_sequences = 0
                    for event_id, count in timestep_counts:
                        logger.info(f"Event ID: {event_id}, Timestep count: {count}")
                        total_sequences += cell_id_count * (count - self.lag - self.horizon + 1) 
                    
                    steps = max(1, total_sequences // self.batch_size)
                    logger.info(f"Calculated steps for {self.subset}: {steps}")
                    return steps
                
        except Exception as e:
            logger.error(f"Error calculating steps: {e}")
            raise
