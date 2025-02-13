import numpy as np 
import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import pandas as pd
from modules.preprocessor.preprocessor import Preprocessor, get_num_features, feature_columns    
import logging
from datetime import datetime
import sys
import tensorflow as tf

from dotenv import load_dotenv

TRAIN_SUBSET_IDENTIFIER = 'train'
VAL_SUBSET_IDENTIFIER = 'test'
DB_BATCH_SIZE = 10000

logger = logging.getLogger("FloodDataGenerator")
logger.setLevel(logging.INFO)
    
# Configure logging with both file and console handlers
def setup_logger():
    
    logger = logging.getLogger("FloodDataGenerator")
    logger.setLevel(logging.INFO)
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(f'flood_generator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

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
        self.test_subset_event = "Run9"
        self.is_preprocessor = is_preprocessor
        self.pool = SimpleConnectionPool(1, 10, **conn_params)
        self.epochs = epochs
        if is_preprocessor:
           return
        self.preprocessor = Preprocessor()
                # Calculate total number of samples
        try:
            with psycopg2.connect(**conn_params) as conn:
                cursor = conn.cursor()
                if self.subset == VAL_SUBSET_IDENTIFIER:
                    cursor.execute(COUNT_QUERY  + f" WHERE timestep LIKE '%{self.test_subset_event}%'")
                else:
                    cursor.execute(COUNT_QUERY  + f" WHERE timestep NOT LIKE '%{self.test_subset_event}%'")
                total_samples = cursor.fetchone()[0]
                logger.info(f"Total samples calculated: {total_samples}")
        except Exception as e:
            logger.error(f"Error calculating total samples: {e}")
            raise
        self.n_samples = total_samples
    
    def generate_sequences(self):
        try:
            with self.pool.getconn() as conn:
                with conn.cursor('flood_cursor', scrollable=True) as cursor:
                    cursor.itersize = self.batch_size
                    query = f"""
                        SELECT 
                            cell_id, 
                            timestep,
                            SPLIT_PART(timestep, '_', 1) as event,
                            upstream1,
                            upstream2,
                            upstream3,
                            elevation,
                            depth
                        FROM flood_data_light
                        WHERE timestep {'LIKE' if self.subset == 'test' else 'NOT LIKE'} %s
                        ORDER BY cell_id, event, timestep
                    """
                    cursor.execute(query, (f"%{self.test_subset_event}%",))
                    
                    buffer = []
                    current_cell_id = None
                    current_event = None
                    
                    for record in cursor:
                        if current_cell_id != record[0] or current_event != record[2]:
                            if len(buffer) >= self.lag + self.horizon:
                                df = pd.DataFrame(buffer, columns=['cell_id', 'timestep', 'event'] + feature_columns)
                                scaled_features = self.preprocessor.preprocess(df, 
                                                                            self.subset == TRAIN_SUBSET_IDENTIFIER)[feature_columns].values
                                for i in range(len(scaled_features) - self.lag - self.horizon + 1):
                                    x = scaled_features[i:i+self.lag].astype(np.float32)
                                    y = scaled_features[i+self.lag:i+self.lag+self.horizon, -1].astype(np.float32)
                                    yield x, y
                            buffer = []
                            current_cell_id = record[0]
                            current_event = record[2]
                        buffer.append(record)
                    
                    # Process final buffer
                    if len(buffer) >= self.lag + self.horizon:
                        df = pd.DataFrame(buffer, columns=['cell_id', 'timestep', 'event'] + feature_columns)
                        scaled_features = self.preprocessor.preprocess(df, 
                                                                    self.subset == TRAIN_SUBSET_IDENTIFIER)[feature_columns].values
                        for i in range(len(scaled_features) - self.lag - self.horizon + 1):
                            yield (scaled_features[i:i+self.lag].astype(np.float32),
                                  scaled_features[i+self.lag:i+self.lag+self.horizon, -1].astype(np.float32))
                              
        except Exception as e:
            logger.error(f"Error generating sequences: {e}")
            raise
        finally:
            if hasattr(self, 'pool'):
                self.pool.putconn(conn)
    
    def _create_sequences(self, buffer):
        if len(buffer) < self.lag + self.horizon:
            return
            
        df = pd.DataFrame(buffer, columns=['id', 'cell_id', 'timestep', 'elevation', 'x', 'y',
                                        'upstream1', 'upstream2', 'upstream3', 'depth'])
        #want to extract the event from the timestep column. Event is the string before the number "Run1_1" -> "Run1"
        df['event'] = df['timestep'].apply(lambda x: x.split('_')[0])
        for cell_id in df['cell_id'].unique():
            for event in df['event'].unique():
                cell_data = df[(df['cell_id'] == cell_id) & (df['event'] == event)]
                logger.info(cell_data)
                scaled_df = self.preprocessor.preprocess(cell_data, self.subset == TRAIN_SUBSET_IDENTIFIER)
                scaled_features = scaled_df[feature_columns].values
                num_sequences = len(scaled_features) - self.lag - self.horizon + 1
                for i in range(num_sequences):
                    x = scaled_features[i:i+self.lag]
                    y = scaled_features[i+self.lag:i+self.lag+self.horizon, -1]
                    if x.shape[0] == self.lag and y.shape[0] == self.horizon:
                        yield x.astype(np.float32), y.astype(np.float32)
    
    def create_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self.generate_sequences,
            output_signature=(
                tf.TensorSpec(shape=(self.lag, get_num_features()), dtype=tf.float32),
                tf.TensorSpec(shape=(self.horizon,), dtype=tf.float32)
            )
        )
        return (dataset
                .cache()
                .batch(self.batch_size)
                .repeat(self.epochs)
                .prefetch(tf.data.AUTOTUNE))
        
    def __len__(self):
        """Calculate total number of batches"""
        logger.info(f"Calculating number of batches for {self.subset} subset")
        total_sequences = self.n_samples - self.lag - self.horizon + 1
        return max(1, total_sequences // self.batch_size)
    
    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.closeall()
            
    def calculate_steps(self):
        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    # Get unique cell count
                    cur.execute("""
                        SELECT COUNT(DISTINCT cell_id) 
                        FROM flood_data_light
                        WHERE timestep {} '%Run9%'
                    """.format('LIKE' if self.subset == 'test' else 'NOT LIKE'))
                    num_cells = cur.fetchone()[0]
                    
                    # Get timesteps per cell
                    cur.execute("""
                        SELECT COUNT(DISTINCT timestep) 
                        FROM flood_data_light
                        WHERE timestep {} '%Run9%'
                    """.format('LIKE' if self.subset == 'test' else 'NOT LIKE'))
                    num_timesteps = cur.fetchone()[0]
                    
                    # Calculate steps
                    sequences_per_cell = num_timesteps - self.lag - self.horizon + 1
                    total_sequences = num_cells * sequences_per_cell
                    steps = max(1, total_sequences // self.batch_size)
                    logger.info(f"Calculated steps for {self.subset}: {steps}")
                    return steps
        except Exception as e:
            logger.error(f"Error calculating steps: {e}")
            raise
