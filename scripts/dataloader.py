import numpy as np 
import os
import psycopg2
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
    def __init__(self, batch_size, lag, horizon, subset):
        logger.info(f"Initializing FloodDataGenerator: batch_size={batch_size}, lag={lag}, horizon={horizon}, subset={subset}")
        self.batch_size = batch_size
        self.lag = lag
        self.horizon = horizon
        self.test_subset_event = "Run9"
        self.data = self.stream_spatial_data()
        self.subset = subset
        self.num_features = None
        total_samples = 0
        
        # Calculate total number of samples
        try:
            with psycopg2.connect(**conn_params) as conn:
                cursor = conn.cursor()
                if self.subset == 'test':
                    cursor.execute(COUNT_QUERY  + f" WHERE timestep NOT LIKE '%{self.test_subset_event}%'")
                else:
                    cursor.execute(COUNT_QUERY  + f" WHERE timestep NOT LIKE '%{self.test_subset_event}%'")
                total_samples = cursor.fetchone()[0]
                logger.info(f"Total samples calculated: {total_samples}")
        except Exception as e:
            logger.error(f"Error calculating total samples: {e}")
            raise
        self.n_samples = total_samples
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

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
        x = x.reshape(-1, self.lag, self.num_features)
        y = y.reshape(-1, self.horizon)
        # Reshape X for LSTM (samples, timesteps, features)s
        logger.info(f"Batch {idx} processed with X shape: {x.shape}, y shape: {y.shape}")
        return x, y

    def on_epoch_end(self):
        logger.info("Epoch ended, reinitializing data stream")
        self.data = self.stream_spatial_data()

    def stream_spatial_data(self):
        """
        Stream spatial data in chunks using LIMIT/OFFSET for efficient database access
        
        Args:
            subset (str): Dataset subset to stream ('train' or 'test')
            batch_size (int): Number of rows to fetch per batch
            
        Yields:
            pandas.DataFrame: Chunk of spatial data containing:
                - cell_id: Unique identifier for spatial cell
                - timestep: Temporal identifier
                - elevation: Ground elevation
                - x, y: Spatial coordinates
                - upstream1-3: Upstream condition values
                - depth: Water depth
        """
        if self.subset not in [TRAIN_SUBSET_IDENTIFIER, VAL_SUBSET_IDENTIFIER]:
            raise ValueError("subset must be either 'train' or 'test'")
        
        try:
            with psycopg2.connect(**conn_params) as conn:
                query = f"""
                SELECT cell_id, timestep, elevation, ST_X(geom) as x, ST_Y(geom) as y, 
                    upstream1, upstream2, upstream3, depth
                FROM simulation_data 
                JOIN elevation_data USING (cell_id) 
                JOIN upstream_conditions USING (timestep)
                """
                if self.subset == VAL_SUBSET_IDENTIFIER:
                    query += f" WHERE timestep LIKE '%{self.test_subset_event}%'"
                else:
                    query += f" WHERE timestep NOT LIKE '%{self.test_subset_event}%'"
                    
                query += "ORDER BY cell_id, timestep"
                
                chunks = pd.read_sql(query, conn, chunksize=self.batch_size) 
                yield from chunks
                    
        except Exception as e:
            logger.error(f"Error streaming data: {e}")
            raise

