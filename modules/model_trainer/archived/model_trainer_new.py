import psycopg2
import pandas as pd
import os
import tensorflow as tf

from dotenv import load_dotenv;
load_dotenv()

conn = psycopg2.connect(
    dbname='carlisle_flood',
    user= 'postgres',
    password= os.getenv('POSTGRES_PASSWORD'),
    host= 'localhost',
)

def fetch_data_in_chunks(chunk_size=100000):
    query = "SELECT * FROM your_table LIMIT %s OFFSET %s"
    offset = 0
    while True:
        df = pd.read_sql(query, conn, params=(chunk_size, offset))
        if df.empty:
            break
        yield df.values
        offset += chunk_size
        
def preprocess_data(batch):
    # Your preprocessing code here
    return batch
        
def tf_data_generator():
    for batch in fetch_data_in_chunks(100000):
        batch = preprocess_data(batch)
        X = batch[:, :-1]  # Features
        y = batch[:, -1]   # Target
        yield X, y

# Convert generator into a tf.data.Dataset
dataset = tf.data.Dataset.from_generator(
    tf_data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, X.shape[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

# Batch and prefetch for performance
dataset = dataset.batch(64).prefetch(tf.data.AUTOTUNE)

