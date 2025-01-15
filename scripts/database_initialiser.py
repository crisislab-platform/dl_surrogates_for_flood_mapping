import os

import numpy as np
import pandas as pd
import psycopg2
import rasterio as rio
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from shapely.geometry import Point

load_dotenv()
conn_params = {
    'dbname': 'carlisle_flood',
    'user': 'postgres',
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': 'localhost'
}

# Create separate tables for upstream conditions, elevation and simulation data
create_upstream_table_query = '''
    CREATE TABLE IF NOT EXISTS upstream_conditions (
        id SERIAL PRIMARY KEY,
        timestep VARCHAR(255),
        upstream1 FLOAT,
        upstream2 FLOAT,
        upstream3 FLOAT
    )
'''

create_elevation_table_query = '''
    CREATE TABLE IF NOT EXISTS elevation_data (
        id SERIAL PRIMARY KEY,
        geom GEOMETRY(Point,27700),
        cell_id VARCHAR(255),
        x_coordinate FLOAT,
        y_coordinate FLOAT,
        elevation FLOAT
    )
'''

create_simulation_table_query = '''
    CREATE TABLE IF NOT EXISTS simulation_data (
        id SERIAL PRIMARY KEY,
        cell_id VARCHAR(255),
        depth FLOAT,
        timestep VARCHAR(255)
    )
'''

drop_table_upstream = ''' DROP TABLE IF EXISTS upstream_conditions'''
drop_table_elevation = ''' DROP TABLE IF EXISTS elevation_data'''
drop_table_simulation = ''' DROP TABLE IF EXISTS simulation_data'''

# Enable PostGIS extension
enable_postgis_query = "CREATE EXTENSION IF NOT EXISTS postgis"

elevation_file_path = '/home/91/23016891/projects/Rapid_FloodModelling_CNN/Data/Carlisle_5m.asc'
bc_data_dir = "/home/91/23016891/projects/Rapid_FloodModelling_CNN/Data/"
lisflood_simulation_dir = "/home/91/23016891/projects/Rapid_FloodModelling_CNN/Data/DEM5m_2D/"

def load_elevation_data():
    
    with rio.open(elevation_file_path) as src:
        # Read the data
        data = src.read(1)
        transform = src.transform
        rows, cols = np.indices(data.shape)
        rows = rows.flatten()
        cols = cols.flatten()
        id = [f"{r}_{c}" for r, c in zip(rows, cols)]
        xs, ys = rio.transform.xy(transform, rows, cols, offset='center')
        xs_flat = np.array(xs).flatten()
        ys_flat = np.array(ys).flatten()
        data_flat = data.flatten()
        df = pd.DataFrame({
            'cell_id': id,
            'x_coordinate': xs_flat,
            'y_coordinate': ys_flat,
            'elevation': data_flat
        })
    return df

def load_data_batches(batch_size=10):
    print('Loading LISFLOOD data in batches')
    ## Initial Boundry Condition Data
    csv_files = [file for file in os.listdir(bc_data_dir) if file.endswith('.csv')]
    csv_files.sort()

    ## LISFLOOD Simulation Data
    inundation_files = {}
    for i in range(1, 3):
        inun_files = [file for file in os.listdir(lisflood_simulation_dir) if file.endswith('.wd') and file.startswith(f"Run{i}")]
        inundation_files[f"Run{i}"] = inun_files if len(inun_files) > 0 else []
    inun_files.sort()

    print(inundation_files)

    # Elevation Data
    elevation_df = load_elevation_data()
    elevation_df = elevation_df.set_index('cell_id')

    for key, inun_files in inundation_files.items():
        if len(inun_files) == 0:
            continue
        bc_file_name = [filename for filename in csv_files if key in filename][0]
        bc_df = pd.read_csv(os.path.join(bc_data_dir, bc_file_name))
        bc_df['timestep'] = ["Run" + str(csv_files.index(bc_file_name) + 1) + "_" + str(idx) for idx in bc_df.index]
        bc_df = bc_df.set_index('timestep')
        #Drop first 8 rows
        bc_df = bc_df[8:]

        # inun_files = inun_files[8:] 
        for i in range(0, len(inun_files), batch_size):
            batch_files = inun_files[i:i + batch_size]
            if len(batch_files) == 0:
                continue
            batch_data = []
            for file in batch_files:
                timestep = key + "_" + str(i)
                data = rio.open(os.path.join(lisflood_simulation_dir, file))
                values = data.read(1)
                rows, cols = np.indices(values.shape)
                rows = rows.flatten()
                cols = cols.flatten()
                depth = values.flatten()
                cell_ids = [f"{r}_{c}" for r, c in zip(rows, cols)]
                df = pd.DataFrame({'cell_id': cell_ids, 'depth': depth})
                df['timestep'] = timestep
                df.loc[df['depth'] < 0.3, 'depth'] = 0
                batch_data.append(df)
            target_df = pd.concat(batch_data, axis=0, ignore_index=True)
            print(target_df.shape)
            print(bc_df.shape)
            target_df['Upstream1'] = target_df['timestep'].map(bc_df['Upstream1'])
            target_df['Upstream2'] = target_df['timestep'].map(bc_df['Upstream2'])
            target_df['Upstream3'] = target_df['timestep'].map(bc_df['Upstream3'])
            target_df['elevation'] = target_df['cell_id'].map(elevation_df['elevation'])
            target_df['x_coordinate'] = target_df['cell_id'].map(elevation_df['x'])
            target_df['y_coordinate'] = target_df['cell_id'].map(elevation_df['y'])
            yield target_df
            

def db_execute(query):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    res = cur.execute(query)
    conn.commit()
    conn.close()
    return res
    

def create_database(db_name):
    """Create a new database with autocommit enabled"""
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        res = cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Database {db_name} created successfully")
        return res
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")
    finally:
        cur.close()
        conn.close()

def execute_many(query, data):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.executemany(query, data)
    conn.commit()
    conn.close()

def execute_batch(query, data):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    execute_values(cur, query, data)
    conn.commit()
    conn.close()

# Insert data into elevation table
def insert_elevation_data():
    elevation_df = load_elevation_data()
    query = '''
        INSERT INTO elevation_data (geom, cell_id, x_coordinate, y_coordinate, elevation)
        VALUES %s;
    '''
    data = elevation_df.copy()
    data['geom'] = data.apply(lambda x: Point(x['x_coordinate'], x['y_coordinate']).wkt, axis=1)
    data = data[['geom', 'cell_id', 'x_coordinate', 'y_coordinate', 'elevation']]
    data = data.values
    # template="(ST_GeomFromText(%s, 27700), %s, %s, %s, %s)"
    # data = [(f"ST_GeomFromText({row[0]}, 27700)", row[1], row[2], row[3], row[4]) for row in data]
    execute_batch(query, data)

# insert data into upstream conditions table
def insert_upstream_data():
    csv_files = [file for file in os.listdir(bc_data_dir) if file.endswith('.csv')]
    csv_files.sort()
    query = '''
        INSERT INTO upstream_conditions (timestep, upstream1, upstream2, upstream3)
        VALUES (%s, %s, %s, %s);
    '''
    for idx, file in enumerate(csv_files):
        print(f"Processing file {idx}")
        bc_df = pd.read_csv(os.path.join(bc_data_dir, file))
        bc_df['timestep'] = ["Run" + str(csv_files.index(file) + 1) + "_" + str(idx) for idx in bc_df.index]
        bc_df = bc_df.set_index('timestep')
        bc_df = bc_df[['Upstream1', 'Upstream2', 'Upstream3']]
        bc_df = bc_df.dropna()
        bc_df = bc_df.reset_index()
        bc_df = bc_df.values
        print(len(bc_df))
        execute_many(query, bc_df)

def insert_simulation_data():
    inundation_files = {}
    for i in range(1, 3):
        inun_files = [file for file in os.listdir(lisflood_simulation_dir) if file.endswith('.wd') and file.startswith(f"Run{i}")]
        inundation_files[f"Run{i}"] = inun_files if len(inun_files) > 0 else []
    inun_files.sort()

    query = '''
        INSERT INTO simulation_data (cell_id, depth, timestep)
        VALUES %s
    '''

    for key, inun_files in inundation_files.items():
        if len(inun_files) == 0:
            continue

        for i in range(0, len(inun_files)):
            print(f"Processing file {i}")
            file = inun_files[i]
            timestep = key + "_" + str(i)
            data = rio.open(os.path.join(lisflood_simulation_dir, file))
            values = data.read(1)
            rows, cols = np.indices(values.shape)
            rows = rows.flatten()
            cols = cols.flatten()
            depth = values.flatten()
            cell_ids = [f"{r}_{c}" for r, c in zip(rows, cols)]
            df = pd.DataFrame({'cell_id': cell_ids, 'depth': depth})
            df['timestep'] = timestep
            df.loc[df['depth'] < 0.3, 'depth'] = 0
            df = df.dropna()
            df = df[['cell_id', 'depth', 'timestep']]
            df = df.values
            print(df.shape)
            execute_batch(query, df)

if __name__ == '__main__':
    # create_database('carlisle_flood')
    # db_execute(enable_postgis_query)
    # db_execute(drop_table_upstream)
    # db_execute(drop_table_elevation)
    # db_execute(drop_table_simulation)
    # db_execute(create_upstream_table_query)
    # db_execute(create_elevation_table_query)
    # db_execute(create_simulation_table_query)
    # insert_elevation_data()
    insert_upstream_data()
    insert_simulation_data()

