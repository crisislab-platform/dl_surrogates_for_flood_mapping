import pandas as pd
import sklearn.preprocessing as preprocessing
from lib.constants import TRAIN_SUBSET_IDENTIFIER, VAL_SUBSET_IDENTIFIER, TEST_EVENT_ID
import logging
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("DataPreprocessor")
feature_columns = ["upstream1", "upstream2", "upstream3", "elevation", "depth"]

def get_num_features() -> int:
        return len(feature_columns) # exclude cell_id, timestep, and coordinate columns

conn_params = {
    'dbname': 'carlisle_flood',
    'user': 'postgres',
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': 'localhost'
}

class Preprocessor:
    def __init__(self):
        logger.info("Initialising Preprocessor")
        self.train_scaler = preprocessing.StandardScaler()
        self.test_scaler = preprocessing.StandardScaler()
        self._init_scalers()
        
    def _init_scalers(self):
        try:
            with psycopg2.connect(**conn_params) as conn:
                logger.info("Initialising scalers")
                
                # Training data scaler
                train_query = f"""
                SELECT upstream1, upstream2, upstream3, elevation, depth
                FROM flood_data_light 
                WHERE event_id != {TEST_EVENT_ID}
                ORDER BY RANDOM()
                LIMIT 100000
                """
                train_data = pd.read_sql(train_query, conn)
                self.train_scaler.fit(train_data[feature_columns])
                logger.info(f"Train scaler mean: {self.train_scaler.mean_}")
                
                # Test data scaler
                test_query = f"""
                SELECT upstream1, upstream2, upstream3, elevation, depth
                FROM flood_data_light
                WHERE event_id = {TEST_EVENT_ID}
                ORDER BY RANDOM()
                LIMIT 100000
                """
                test_data = pd.read_sql(test_query, conn)
                self.test_scaler.fit(test_data[feature_columns])
                logger.info(f"Test scaler mean: {self.test_scaler.mean_}")
                
        except Exception as e:
            logger.error(f"Error initializing scalers: {e}")
            raise
        
    def preprocess(self, dataframe:pd.DataFrame, is_train:bool) -> pd.DataFrame:
        #Scale the data
        if is_train:
            self.scaler = self.train_scaler.transform(dataframe[feature_columns])
        if not is_train:
            self.scaler = self.test_scaler.transform(dataframe[feature_columns])
        return dataframe