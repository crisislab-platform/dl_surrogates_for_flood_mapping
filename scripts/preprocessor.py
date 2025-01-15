import pandas as pd
import sklearn.preprocessing as preprocessing
import logging

logger = logging.getLogger("DataPreprocessor")

feature_columns = ["upstream1", "upstream2", "upstream3", "depth", "elevation"]

def preprocess_carlisle_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data")

    # Scale the data
    
    scaler = preprocessing.StandardScaler()
    dataframe[feature_columns] = scaler.fit_transform(dataframe[feature_columns])
    
    logger.info("Finished preprocessing data")
    return dataframe
    
def get_num_features() -> int:
    return len(feature_columns) # exclude cell_id, timestep, and coordinate and elevation columns
        
    
    