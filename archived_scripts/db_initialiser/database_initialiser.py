import os
import numpy as np
import pandas as pd
import psycopg2
import rasterio as rio
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from shapely.geometry import Point
from modules.utils.path_util import PROJECT_ROOT

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatabaseInitialiser")

elevation_file_path = f'{PROJECT_ROOT}/carlisle-data/Carlisle_5m.asc'
bc_data_dir = f"{PROJECT_ROOT}/carlisle-data/"
lisflood_simulation_dir = f"{PROJECT_ROOT}/carlisle-data/DEM5m_2D/"

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

create_materialised_view_query = '''
CREATE MATERIALIZED VIEW flood_data_mv AS
SELECT 
    s.cell_id,
    s.timestep,
    e.elevation,
    ST_X(e.geom) as x,
    ST_Y(e.geom) as y,
    u.upstream1,
    u.upstream2,
    u.upstream3,
    s.depth
FROM 
    simulation_data s
    JOIN elevation_data e USING (cell_id)
    JOIN upstream_conditions u USING (timestep)
ORDER BY cell_id, timestep;
'''

create_light_weight_table_query = '''
CREATE TABLE IF NOT EXISTS flood_data_light (
    id SERIAL PRIMARY KEY,
    event_id int,
    cell_id int,
    timestep int,
    elevation FLOAT,
    x FLOAT,
    y FLOAT,
    upstream1 FLOAT,
    upstream2 FLOAT,
    upstream3 FLOAT,
    depth FLOAT
)
'''

create_indexes_query = '''
CREATE INDEX idx_flood_mv_cell_time ON flood_data_mv(cell_id, timestep);
CREATE INDEX idx_flood_mv_timestep ON flood_data_mv(timestep);
'''

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

def load_elevation_data(window):
    with rio.open(elevation_file_path) as src:
        # Read the data
        data = src.read(1, window=window)
        transform = src.transform
        rows, cols = np.indices(data.shape)
        rows = rows.flatten() + window.row_off
        cols = cols.flatten() + window.col_off
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
        #Drop first 8 rows as the first 8 timesteps are taken for initialisation of LISFLOOD simulation
        bc_df = bc_df[8:]

        # Drop frist 8 simulation files as 2 hours (8 timesteps are taken for initialisation of LISFLOOD simulation)
        inun_files = inun_files[8:]
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
        bc_df = bc_df[8:] # Drop first 8 rows as the first 8 timesteps are taken for initialisation of LISFLOOD simulation
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
    for i in range(1, 10):
        inun_files = [file for file in os.listdir(lisflood_simulation_dir) if file.endswith('.wd') and file.startswith(f"Run{i}")]
        inundation_files[f"Run{i}"] = inun_files if len(inun_files) > 0 else []

    query = '''
        INSERT INTO simulation_data (cell_id, depth, timestep)
        VALUES %s
    '''
    for key, inun_files in inundation_files.items():
        if len(inun_files) == 0:
            continue

        inun_files.sort()
        inun_files = inun_files[8:] # Drop first 8 simulation files as 2 hours (8 timesteps are taken for initialisation of LISFLOOD simulation)
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
            
def load_inundation_data(window):
    inundation_files = {}
    
    for i in range(1, 10):
        inun_files = [file for file in os.listdir(lisflood_simulation_dir) if file.endswith('.wd') and file.startswith(f"Run{i}")]
        inun_files.sort()
        inun_files = inun_files[8:] # Drop first 8 simulation files as 2 hours (8 timesteps are taken for initialisation of LISFLOOD simulation)
        inundation_files[f"Run{i}"] = inun_files if len(inun_files) > 0 else []
        
    final_df = pd.DataFrame()
    for key, inun_files in inundation_files.items():
        if len(inun_files) == 0:
            continue
        for i in range(0, len(inun_files)):
            file = inun_files[i]
            timestep = key + "_" + str(i)
            with rio.open(os.path.join(lisflood_simulation_dir, file)) as src:
                values = src.read(1, window=window)
                # Create indices relative to the window
                rows, cols = np.indices(values.shape)
                rows = rows.flatten() + window.row_off
                cols = cols.flatten() + window.col_off
                depth = values.flatten()
                cell_ids = [f"{r}_{c}" for r, c in zip(rows, cols)]
                df = pd.DataFrame({
                    'cell_id': cell_ids,
                    'depth': depth,
                    'timestep': timestep
                })
                final_df = pd.concat([final_df, df], axis=0, ignore_index=True)
    
    return final_df

def load_upstream_data():
    csv_files = [file for file in os.listdir(bc_data_dir) if file.endswith('.csv')]
    csv_files.sort()

    final_bc_df = pd.DataFrame()
    for bc_file_name in csv_files:
        bc_df = pd.read_csv(os.path.join(bc_data_dir, bc_file_name))
        bc_df = bc_df[8:]
        bc_df.reset_index(inplace=True)
        bc_df['timestep'] = ["Run" + str(csv_files.index(bc_file_name) + 1) + "_" + str(idx) for idx in bc_df.index]
        #Drop first 8 rows as the first 8 timesteps are taken for initialisation of LISFLOOD simulation

        final_bc_df = pd.concat([final_bc_df, bc_df], axis=0, ignore_index=True)
        
    return final_bc_df

def get_random_window(window_length = None):
    with rio.open(elevation_file_path) as raster:
        rows = raster.height
        cols = raster.width
        
        max_row = rows - window_length
        max_col = cols - window_length
    
        # Generate valid random coordinates
        random_row = np.random.randint(0, max_row) if max_row > 0 else 0
        random_col = np.random.randint(0, max_col) if max_col > 0 else 0
    
        window = rio.windows.Window(
            col_off=random_col,
            row_off=random_row,
            width=min(window_length, cols - random_col),
            height=min(window_length, rows - random_row)
        )
        return window
    
def get_window_batches(batch_size=50):
    """Generate windows to cover the entire raster without missing any cells"""
    with rio.open(elevation_file_path) as raster:
        total_rows = raster.height
        total_cols = raster.width
        
        # Calculate number of full windows needed
        n_rows = (total_rows + batch_size - 1) // batch_size
        n_cols = (total_cols + batch_size - 1) // batch_size
        
        logger.info(f"Processing raster of size {total_rows}x{total_cols} in {n_rows}x{n_cols} windows")
        
        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate actual window dimensions
                row_start = row * batch_size
                col_start = col * batch_size
                row_end = min(row_start + batch_size, total_rows)
                col_end = min(col_start + batch_size, total_cols)
                
                window = rio.windows.Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=col_end - col_start,
                    height=row_end - row_start
                )
                
                logger.info(f"Generated window: rows {row_start}:{row_end}, cols {col_start}:{col_end}")
                yield window
    
    
def insert_data(window):
    # Load elevation data for above window
    elevation_df = load_elevation_data(window)
    logger.info(f"Elevation data loaded successfully, size: {elevation_df.shape}")
    inundation_data = load_inundation_data(window)
    logger.info(f"Inundation data loaded successfully, size: {inundation_data.shape}")
    upstream_data = load_upstream_data()
    logger.info(f"Upstream data loaded successfully, size: {upstream_data.shape}")
        
    # merge the data based on timestep and cell_id
    #convert timestep to string
    inundation_data['timestep'] = inundation_data['timestep'].astype(str)
    upstream_data['timestep'] = upstream_data['timestep'].astype(str)
    
    logger.info(inundation_data.head())
    logger.info(upstream_data.head())
    logger.info(elevation_df.head())
    
    merged_df = pd.merge(inundation_data, upstream_data, on='timestep')
    logger.info(f"Merged inundation and upstream data successfully, size: {merged_df.shape}")

    # convert cell_id to string
    merged_df['cell_id'] = merged_df['cell_id'].astype(str)
    elevation_df['cell_id'] = elevation_df['cell_id'].astype(str)
    merged_df = pd.merge(merged_df, elevation_df, on='cell_id')
    
    logger.info(f"Merged elevation data with merged data")
    merged_df = merged_df[['cell_id', 'timestep', 'elevation', 'x_coordinate', 'y_coordinate', 'Upstream1', 'Upstream2', 'Upstream3', 'depth']]
    
    merged_df['event_id'] = merged_df['timestep'].apply(lambda x: int(x.split('_')[0].replace('Run','')))
    merged_df['timestep'] = merged_df['timestep'].apply(lambda x: int(x.split('_')[1]))
    
    # Reorder columns to match new schema
    merged_df = merged_df[['event_id', 'cell_id', 'timestep', 'elevation', 'x_coordinate', 'y_coordinate',
                           'Upstream1', 'Upstream2', 'Upstream3', 'depth']]
    
    logger.info(f"Final merged data size: {merged_df.shape}")
    logger.info(merged_df.head())
    
    # Insert data into the light weight table
    query = '''
        INSERT INTO flood_data_light (
            event_id, cell_id, timestep,
            elevation, x, y,
            upstream1, upstream2, upstream3, depth
        )
        VALUES %s
    '''
    logger.info("Inserting data into light weight table")
    execute_batch(query, merged_df.values)
    logger.info("Data inserted successfully")
            
def insert_light_weight_table(window_length = None):
    logger.info("Inserting data into light weight table")

    #If window length is not provided, use the entire raster, but have to do it in batches
    if window_length is None or window_length <= 0:
        # Process windows of size 50 x 50 for the entire raster
        for window in get_window_batches(50):  # Remove enumerate() as we don't need the counter
            logger.info(f"Processing window: {window}")
            insert_data(window)
    
    else:
        window = get_random_window(window_length)
        logger.info(f"Random window created of size: {window}")
        insert_data(window)
        
def create_materialised_view():
    db_execute(create_materialised_view_query)
    db_execute(create_indexes_query)
    
def create_light_weight_table():
    db_execute(create_light_weight_table_query)
    logger.info("Light weight table created successfully")  
    
def drop_table(table_name):
    db_execute(f"DROP TABLE IF EXISTS {table_name}")
    logger.info(f"Table {table_name} dropped successfully")


def init(window_length = None):
    #create_database('carlisle_flood')
    # db_execute(enable_postgis_query)
    # db_execute(drop_table_upstream)
    # # db_execute(drop_table_elevation)
    # db_execute(drop_table_simulation)
    # db_execute(create_upstream_table_query)
    # # db_execute(create_elevation_table_query)
    # db_execute(create_simulation_table_query)
    # # insert_elevation_data()
    # insert_upstream_data()
    # insert_simulation_data()
    
    # create indexes for faster query
    # db_execute("CREATE INDEX cell_id_idx ON elevation_data(cell_id)")
    # db_execute("CREATE INDEX timestep_idx ON upstream_conditions(timestep)")
    # db_execute("CREATE INDEX timestep_idx ON simulation_data(timestep)")
    # db_execute("CREATE INDEX cell_id_idx ON simulation_data(cell_id)")
    # create_light_weight_table()
    drop_table("flood_data_light")
    create_light_weight_table()
    insert_light_weight_table(window_length)
