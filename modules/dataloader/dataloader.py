import numpy as np 
import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.utils import Sequence
from preprocessor import preprocess_carlisle_data
import logging
from datetime import datetime
import sys

TRAIN_SUBSET_IDENTIFIER = 'train'
VAL_SUBSET_IDENTIFIER = 'test'

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

logger = setup_logger()

load_dotenv()

conn_params = {
    'dbname': 'carlisle_flood',
    'user': 'postgres',
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': 'localhost'
}

COUNT_QUERY = """
                SELECT COUNT(*) FROM simulation_data 
                JOIN elevation_data USING (cell_id) 
                JOIN upstream_conditions USING (timestep)
            """

class FloodDataGenerator(Sequence):
    """
    Data generator for flood forecasting with temporal sequences
    """
    _pool = None  # Class-level connection pool
    
    @classmethod
    def get_pool(cls):
        if cls._pool is None:
            cls._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **conn_params
            )
        return cls._pool
    
    def __init__(self, batch_size, lag, horizon, subset):
        logger.info(f"Initializing FloodDataGenerator: batch_size={batch_size}, lag={lag}, horizon={horizon}, subset={subset}")
        self.batch_size = batch_size
        self.lag = lag
        self.horizon = horizon
        self.test_subset_event = "Run9"
        self.data = self.stream_spatial_data()
        self.subset = subset
        self.num_features = None
        self.pool = self.get_pool()
        total_samples = 0
        
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
    
    def __len__(self):
        """Calculate total number of batches"""
        total_sequences = self.n_samples - self.lag - self.horizon + 1
        return max(1, total_sequences // self.batch_size)

    def __getitem__(self, idx):
        logger.info(f"Fetching batch {idx}")
        # Load specific chunk from database
        while self.data is None:
            logger.error("Error loading data, data is None")
            
        chunk = next(self.data)
        logger.info(f"Chunk loaded with shape: {chunk.shape}")
        preporcessed_data = preprocess_carlisle_data(chunk)
        logger.info(f"Data preprocessed with shape: {preporcessed_data.shape}")
        
        if self.num_features is None:
            self.num_features = preporcessed_data.shape[1] - 4 # exclude cell_id, timestep, and coordinates
        
        # Prepare X and y
        x,y = [], []
        
        logger.info(f"Creating sequences with lag={self.lag} and horizon={self.horizon} for batch {idx}")
        for cell_id in preporcessed_data['cell_id'].unique():
            
            cell_data = preporcessed_data[preporcessed_data['cell_id'] == cell_id]
            # drop cell_id, timestep, and coordinate columns
            cell_data = cell_data.drop(columns=['cell_id', 'timestep', 'x', 'y'])
            for i in range(cell_data.shape[0] - self.lag):
                x.append(cell_data.iloc[i:i+self.lag].values)
                y.append(cell_data.iloc[i+self.lag:i+self.lag + self.horizon]['depth'].values)
        
        # x already has the shape (samples, timesteps, features) just need to convert to numpy array
        x = np.array(x)
        y = np.array(y)
        logger.info(f"X shape: {x.shape}, y shape: {y.shape}")
        x = x.reshape(-1, self.lag, self.num_features)
        y = y.reshape(-1, self.horizon)
        # Reshape X for LSTM (samples, timesteps, features)
        logger.info(f"Batch {idx} reshaped with X shape: {x.shape}, y shape: {y.shape}")
        return x, y

    def on_epoch_end(self):
        logger.info("Epoch ended, reinitializing data stream")
        self.data = self.stream_spatial_data()

    def stream_spatial_data(self):
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor('flood_data_cursor') as cursor:
                # Base query with window function for improved performance
                query = """
                WITH numbered_rows AS (
                    SELECT 
                        cell_id, 
                        timestep, 
                        elevation::float4, 
                        ST_X(geom)::float4 as x, 
                        ST_Y(geom)::float4 as y,
                        upstream1::float4, 
                        upstream2::float4, 
                        upstream3::float4, 
                        depth::float4,
                        ROW_NUMBER() OVER (
                            PARTITION BY cell_id 
                            ORDER BY timestep
                        ) as row_num
                    FROM simulation_data 
                    JOIN elevation_data USING (cell_id) 
                    JOIN upstream_conditions USING (timestep)
                    WHERE timestep {condition} %s
                )
                SELECT *
                FROM numbered_rows
                ORDER BY cell_id, row_num
                """
                
                # Prepare condition and parameters
                condition = "LIKE" if self.subset == VAL_SUBSET_IDENTIFIER else "NOT LIKE"
                params = (f"%{self.test_subset_event}%",)
                
                # Use server-side cursor for memory efficiency
                cursor.itersize = self.batch_size
                cursor.execute(query.format(condition=condition), params)
                
                while True:
                    records = cursor.fetchmany(self.batch_size)
                    if not records:
                        break
                    
                    # Create DataFrame without dtype specification
                    df = pd.DataFrame.from_records(
                        records,
                        columns=['cell_id', 'timestep', 'elevation', 'x', 'y', 
                                'upstream1', 'upstream2', 'upstream3', 'depth', 'row_num']
                    )
                    
                    # Convert types after creation
                    float_columns = ['elevation', 'x', 'y', 'upstream1', 
                                   'upstream2', 'upstream3', 'depth']
                    df[float_columns] = df[float_columns].astype('float32')
                    
                    yield df.drop('row_num', axis=1)
                    
        except Exception as e:
            logger.error(f"Error streaming data: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
                
    def __del__(self):
        """Cleanup connection pool when instance is destroyed"""
        if self._pool:
            self._pool.closeall()
            self.__class__._pool = None

