import os
import h5py
import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from modules.utils.path_util import PROJECT_ROOT
import gc
import json

TEST_SUBSET = 1
VALIDATION_SUBSET = 3
                                                             
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FileSequenceGenerator")

# Reuse paths from database_initialiser
elevation_file_path = f'{PROJECT_ROOT}/carlisle-data/Carlisle_5m.asc'
bc_data_dir = f"{PROJECT_ROOT}/carlisle-data/"
lisflood_simulation_dir = f"{PROJECT_ROOT}/carlisle-data/DEM5m_2D/"

def load_elevation_data():
    """Load elevation data from ASC file"""
    with rio.open(elevation_file_path) as src:
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

def load_upstream_data() -> Dict[str, pd.DataFrame]:
    """Load upstream boundary condition data from CSV files"""
    csv_files = [file for file in os.listdir(bc_data_dir) if file.endswith('.csv')]
    csv_files.sort()
    
    upstream_data = {}
    for file in csv_files:
        run_id = int(file.split('Run')[1].split('.')[0])
        bc_df = pd.read_csv(os.path.join(bc_data_dir, file))
        bc_df = bc_df[8:]  # Drop initialization timesteps
        bc_df.reset_index(inplace=True)
        bc_df['timestep'] = bc_df.index
        upstream_data[run_id] = bc_df[['timestep', 'Upstream1', 'Upstream2', 'Upstream3']]
    
    return upstream_data

def load_simulation_files() -> Dict[int, List[str]]:
    """Get sorted simulation files for each run"""
    inundation_files = {}
    for i in range(1, 4):
        files = [f for f in os.listdir(lisflood_simulation_dir) 
                if f.endswith('.wd') and f.startswith(f"Run{i}")]
        if files:
            files.sort()
            files = files[8:]  # Drop initialization timesteps
            inundation_files[i] = files
    return inundation_files

def get_grid_dimensions():
    """Get the grid dimensions from the elevation file"""
    with rio.open(elevation_file_path) as src:
        height = src.height
        width = src.width
    return height, width

def process_run_to_file(run_id, files, bc_data, elevation_grid, height, width, lag, horizon, 
                       output_dir, date_time, split_name):
    """Process a single run and save to its own file"""
    logger.info(f"Processing run {run_id} for {split_name} split")
    
    # Create feature columns for each cell
    num_features = 5  # Upstream1, Upstream2, Upstream3, elevation, depth
    
    # Create file for this run
    filename = f"grid_seq_{date_time}_run{run_id}_{split_name}_lag-{lag}_horizon-{horizon}.h5"
    file_path = os.path.join(output_dir, filename)
    
    with h5py.File(file_path, 'w') as hf:
        # Create datasets
        X_dataset = hf.create_dataset('X', shape=(0, lag, height*width, num_features), 
                              maxshape=(None, lag, height*width, num_features),
                              chunks=(10, lag, min(10000, height*width), num_features),
                              compression='gzip')
        y_dataset = hf.create_dataset('y', shape=(0, height*width, horizon),
                              maxshape=(None, height*width, horizon),
                              chunks=(10, min(10000, height*width), horizon),
                              compression='gzip')
                              
        # Store metadata
        hf.attrs['run_id'] = run_id
        hf.attrs['split'] = split_name
        hf.attrs['grid_height'] = height
        hf.attrs['grid_width'] = width
        hf.attrs['lag'] = lag
        hf.attrs['horizon'] = horizon
        
        # Define batch sizes
        timestep_batch = 20  # Process this many timesteps at once
        seq_batch = 10      # Generate this many sequences at once
        
        # Process in batches of timesteps
        total_timesteps = len(files)
        seq_count = 0
        
        for start_idx in range(0, total_timesteps, timestep_batch):
            end_idx = min(start_idx + timestep_batch + lag + horizon, total_timesteps)
            
            # Skip if we don't have enough timesteps for a sequence
            if end_idx - start_idx < lag + horizon:
                continue
            
            # Load batch of timesteps
            batch_grids = []
            for t_idx in range(start_idx, end_idx):
                # Create grid for this timestep
                grid = np.zeros((height, width, num_features))
                
                # Set upstream values
                bc_values = bc_data[bc_data['timestep'] == t_idx].iloc[0]
                grid[:, :, 0] = bc_values['Upstream1']
                grid[:, :, 1] = bc_values['Upstream2']
                grid[:, :, 2] = bc_values['Upstream3']
                
                # Set elevation (static)
                grid[:, :, 3] = elevation_grid
                
                # Read depth data
                with rio.open(os.path.join(lisflood_simulation_dir, files[t_idx])) as src:
                    depth_data = src.read(1)
                    depth_data[depth_data < 0.3] = 0
                    grid[:, :, 4] = depth_data
                
                batch_grids.append(grid)
            
            batch_grids = np.array(batch_grids)
            
            # Generate sequences from this batch
            max_seq = len(batch_grids) - lag - horizon + 1
            for seq_start in range(0, max_seq, seq_batch):
                seq_end = min(seq_start + seq_batch, max_seq)
                X_batch = []
                y_batch = []
                
                for i in range(seq_start, seq_end):
                    # Input: lag timesteps
                    X = batch_grids[i:i+lag]
                    
                    # Output: next horizon timesteps (depth only)
                    y = batch_grids[i+lag:i+lag+horizon, :, :, 4]
                    
                    # Reshape for LSTM format
                    X_reshaped = X.reshape(lag, height*width, num_features)
                    y_reshaped = y.reshape(horizon, height*width).T
                    
                    X_batch.append(X_reshaped)
                    y_batch.append(y_reshaped)
                
                # Convert batch to arrays
                X_array = np.array(X_batch)
                y_array = np.array(y_batch)
                
                # Append to datasets
                current_size = X_dataset.shape[0]
                new_size = current_size + len(X_batch)
                
                X_dataset.resize(new_size, axis=0)
                y_dataset.resize(new_size, axis=0)
                
                X_dataset[current_size:new_size] = X_array
                y_dataset[current_size:new_size] = y_array
                
                seq_count += len(X_batch)
                logger.info(f"Run {run_id}: Added {len(X_batch)} sequences, total: {seq_count}")
            
            # Free memory
            del batch_grids
            gc.collect()
        
        logger.info(f"Run {run_id}: Completed with {seq_count} sequences")
        return file_path, seq_count

def process_run_to_file_light(run_id, files, bc_data, height, width, lag, horizon, 
                             output_dir, date_time, split_name):
    """Process a single run with only upstream values as features and save to file"""
    logger.info(f"Processing run {run_id} for {split_name} split (upstream-only)")
    
    # Create feature columns for each cell - only upstream values, no elevation or depth
    num_features = 3  # Upstream1, Upstream2, Upstream3
    
    # Create file for this run
    filename = f"grid_seq_light_{date_time}_run{run_id}_{split_name}_lag-{lag}_horizon-{horizon}.h5"
    file_path = os.path.join(output_dir, filename)
    
    with h5py.File(file_path, 'w') as hf:
        # Create datasets - note X shape has changed to only include upstream features
        X_dataset = hf.create_dataset('X', shape=(0, lag, num_features), 
                              maxshape=(None, lag, num_features),
                              chunks=(10, lag, num_features),
                              compression='gzip')
        y_dataset = hf.create_dataset('y', shape=(0, height*width, horizon),
                              maxshape=(None, height*width, horizon),
                              chunks=(10, min(10000, height*width), horizon),
                              compression='gzip')
                              
        # Store metadata
        hf.attrs['run_id'] = run_id
        hf.attrs['split'] = split_name
        hf.attrs['grid_height'] = height
        hf.attrs['grid_width'] = width
        hf.attrs['lag'] = lag
        hf.attrs['horizon'] = horizon
        hf.attrs['features'] = ['Upstream1', 'Upstream2', 'Upstream3']
        
        # Define batch sizes
        timestep_batch = 20  # Process this many timesteps at once
        seq_batch = 10      # Generate this many sequences at once
        
        # Process in batches of timesteps
        total_timesteps = len(files)
        seq_count = 0
        
        for start_idx in range(0, total_timesteps, timestep_batch):
            end_idx = min(start_idx + timestep_batch + lag + horizon, total_timesteps)
            
            # Skip if we don't have enough timesteps for a sequence
            if end_idx - start_idx < lag + horizon:
                continue
            
            # Load batch of timesteps - only need to load depth for target
            # We'll handle upstream values separately since they're not per-grid-cell
            depth_grids = []
            upstream_values = []
            
            for t_idx in range(start_idx, end_idx):
                # Get upstream values for this timestep
                bc_values = bc_data[bc_data['timestep'] == t_idx].iloc[0]
                upstream = [
                    bc_values['Upstream1'],
                    bc_values['Upstream2'],
                    bc_values['Upstream3']
                ]
                upstream_values.append(upstream)
                
                # Only need to read water depth for targets
                if t_idx >= start_idx + lag:
                    with rio.open(os.path.join(lisflood_simulation_dir, files[t_idx])) as src:
                        depth_data = src.read(1)
                        depth_data[depth_data < 0.3] = 0
                        depth_grids.append(depth_data)
            
            upstream_values = np.array(upstream_values)
            depth_grids = np.array(depth_grids) if depth_grids else None
            
            # Generate sequences from this batch - FIX the calculation
            batch_timesteps = min(timestep_batch, end_idx - start_idx - lag - horizon + 1)
            max_seq = max(0, batch_timesteps)
            
            for seq_start in range(0, max_seq, seq_batch):
                seq_end = min(seq_start + seq_batch, max_seq)
                X_batch = []
                y_batch = []
                
                for i in range(seq_start, seq_end):
                    # Input: lag timesteps of upstream values only
                    X = upstream_values[i:i+lag]
                    
                    # Output: next horizon timesteps (depth only)
                    y_depths = depth_grids[i:i+horizon]
                    
                    # Reshape for model format
                    # X remains as is - just lag timesteps of 3 upstream values
                    # y needs to be reshaped to be (cells, horizon)
                    y_reshaped = np.zeros((height*width, horizon))
                    for h in range(horizon):
                        y_reshaped[:, h] = y_depths[h].flatten()
                    
                    X_batch.append(X)
                    y_batch.append(y_reshaped)
                
                # Convert batch to arrays
                X_array = np.array(X_batch)
                y_array = np.array(y_batch)
                
                # Append to datasets
                current_size = X_dataset.shape[0]
                new_size = current_size + len(X_batch)
                
                X_dataset.resize(new_size, axis=0)
                y_dataset.resize(new_size, axis=0)
                
                X_dataset[current_size:new_size] = X_array
                y_dataset[current_size:new_size] = y_array
                
                seq_count += len(X_batch)
                logger.info(f"Run {run_id}: Added {len(X_batch)} upstream-only sequences, total: {seq_count}")
            
            # Free memory
            del upstream_values, depth_grids
            gc.collect()
        
        logger.info(f"Run {run_id}: Completed with {seq_count} upstream-only sequences")
        return file_path, seq_count

def generate_grid_sequences(lag: int, horizon: int, output_dir: str):
    """Generate grid-based sequences from files and save to multiple HDF5 files"""
    logger.info("Loading base data...")
    elevation_df = load_elevation_data()
    upstream_data = load_upstream_data()
    simulation_files = load_simulation_files()
    
    # Get grid dimensions
    height, width = get_grid_dimensions()
    num_cells = height * width
    logger.info(f"Grid dimensions: {height}x{width}, total cells: {num_cells}")
    
    # Split runs into train/validation/test
    train_runs = [run for run in simulation_files.keys() if run not in [TEST_SUBSET, VALIDATION_SUBSET]]
    val_runs = [run for run in simulation_files.keys() if run == VALIDATION_SUBSET]
    test_runs = [run for run in simulation_files.keys() if run == TEST_SUBSET]
    
    splits = {
        'train': train_runs,
        'validation': val_runs,
        'test': test_runs
    }
    
    # Preprocess elevation data into a grid
    elevation_grid = np.zeros((height, width))
    for index, row in tqdm(elevation_df.iterrows(), total=len(elevation_df), desc="Creating elevation grid"):
        r, c = map(int, row['cell_id'].split('_'))
        if 0 <= r < height and 0 <= c < width:
            elevation_grid[r, c] = row['elevation']
    
    # Create output directory
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a metadata file to track all generated files
    metadata_file = os.path.join(output_dir, f"grid_sequences_{date_time}_metadata.json")
    metadata = {
        "date_time": date_time,
        "lag": lag,
        "horizon": horizon,
        "grid_dimensions": {"height": height, "width": width},
        "files": {}
    }
    
    # Process each run and save to separate file
    for split_name, run_ids in splits.items():
        logger.info(f"Processing {split_name} split...")
        metadata["files"][split_name] = []
        
        # Process one event at a time
        for run_id in run_ids:
            files = simulation_files[run_id]
            bc_data = upstream_data[run_id]
            
            # Process this run to its own file
            file_path, seq_count = process_run_to_file(
                run_id, files, bc_data, elevation_grid, 
                height, width, lag, horizon, output_dir, date_time, split_name
            )
            
            # Add to metadata
            metadata["files"][split_name].append({
                "run_id": int(run_id),
                "file_path": file_path,
                "sequences": seq_count
            })
            
            # Force garbage collection
            gc.collect()
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Successfully saved all sequences to {output_dir}")
    logger.info(f"Metadata saved to {metadata_file}")

def generate_grid_sequences_from_files(lag, horizon, light=False):
    if light:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(PROJECT_ROOT, "carlisle-data", "grid_sequential_light")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
     
        output_dir = os.path.join(run_dir, f"grid_sequences_light_{date_time}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        generate_grid_sequences_light(lag=lag, horizon=horizon, output_dir=output_dir)
        return output_dir
    else:
        """Generate grid-based sequences and save to files"""
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(PROJECT_ROOT, "carlisle-data", "grid_sequential")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
    
        output_dir = os.path.join(run_dir, f"grid_sequences_{date_time}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        generate_grid_sequences(lag=lag, horizon=horizon, output_dir=output_dir)
        return output_dir

def generate_grid_sequences_light(lag: int, horizon: int, output_dir: str):
    """Generate grid-based sequences with upstream values only from files and save to HDF5 files"""
    logger.info("Loading base data for upstream-only sequence generation...")
    upstream_data = load_upstream_data()
    simulation_files = load_simulation_files()
    
    # Get grid dimensions
    height, width = get_grid_dimensions()
    num_cells = height * width
    logger.info(f"Grid dimensions: {height}x{width}, total cells: {num_cells}")
    
    # Split runs into train/validation/test
    train_runs = [run for run in simulation_files.keys() if run not in [TEST_SUBSET, VALIDATION_SUBSET]]
    val_runs = [run for run in simulation_files.keys() if run == VALIDATION_SUBSET]
    test_runs = [run for run in simulation_files.keys() if run == TEST_SUBSET]
    
    splits = {
        'train': train_runs,
        'validation': val_runs,
        'test': test_runs
    }
    
    # Create output directory
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a metadata file to track all generated files
    metadata_file = os.path.join(output_dir, f"grid_sequences_light_{date_time}_metadata.json")
    metadata = {
        "date_time": date_time,
        "lag": lag,
        "horizon": horizon,
        "grid_dimensions": {"height": height, "width": width},
        "files": {}
    }
    
    # Process each run and save to separate file
    for split_name, run_ids in splits.items():
        logger.info(f"Processing {split_name} split (upstream-only)...")
        metadata["files"][split_name] = []
        
        # Process one event at a time
        for run_id in run_ids:
            files = simulation_files[run_id]
            bc_data = upstream_data[run_id]
            
            # Process this run to its own file with upstream-only features
            file_path, seq_count = process_run_to_file_light(
                run_id, files, bc_data, height, width, lag, horizon, 
                output_dir, date_time, split_name
            )
            
            # Add to metadata
            metadata["files"][split_name].append({
                "run_id": int(run_id),
                "file_path": file_path,
                "sequences": seq_count
            })
            
            # Force garbage collection
            gc.collect()
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Successfully saved all upstream-only sequences to {output_dir}")
    logger.info(f"Metadata saved to {metadata_file}")
    
    return metadata_file

def generate_grid_sequences_light_from_files(lag, horizon):
    """Generate grid-based sequences with upstream values only and save to files"""
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROJECT_ROOT, "carlisle-data", "grid_sequential_light")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
 
    output_dir = os.path.join(run_dir, f"grid_sequences_light_{date_time}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metadata_file = generate_grid_sequences_light(lag=lag, horizon=horizon, output_dir=output_dir)
    return output_dir, metadata_file
