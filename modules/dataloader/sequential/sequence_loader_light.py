import os
import h5py
import numpy as np
import pandas as pd
import rasterio as rio
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from modules.utils.path_util import PROJECT_ROOT
import gc
import json

TEST_SUBSET = "Run1"
VALIDATION_SUBSET = "Run9"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SequenceLoaderLight")

# Paths from database_initialiser
bc_data_dir = f"{PROJECT_ROOT}/carlisle-data/"
lisflood_simulation_dir = f"{PROJECT_ROOT}/carlisle-data/DEM5m_2D/"

def load_upstream_data() -> Dict[str, pd.DataFrame]:
    """Load upstream boundary condition data from CSV files"""
    csv_files = [file for file in os.listdir(bc_data_dir) if file.startswith('Upstream_Flows') and file.endswith('.csv')]
    csv_files.sort()
    
    upstream_data = {}
    for file in csv_files:
        run_id = file.split('_')[-1].split('.')[0]
        bc_df = pd.read_csv(os.path.join(bc_data_dir, file))
        bc_df = bc_df[8:]  # Drop initialization timesteps
        bc_df.reset_index(inplace=True, drop=True)
        bc_df['timestep'] = bc_df.index
        upstream_data[run_id] = bc_df[['timestep', 'Upstream1', 'Upstream2', 'Upstream3']]
    
    return upstream_data

def load_simulation_files() -> Dict[str, List[str]]:
    """Get sorted simulation files for each run"""
    inundation_files = {}
    for i in range(1, 10):  # Runs 1-9
        run_id = f"Run{i}"
        files = [f for f in os.listdir(lisflood_simulation_dir) 
                if f.endswith('.wd') and f.startswith(run_id)]
        if files:
            files.sort()
            files = files[8:]  # Drop initialization timesteps
            inundation_files[run_id] = files
    return inundation_files

def get_grid_dimensions():
    """Get the grid dimensions from a sample water depth file"""
    sample_file = None
    for file in os.listdir(lisflood_simulation_dir):
        if file.endswith('.wd'):
            sample_file = file
            break
    
    if not sample_file:
        raise ValueError("No water depth files found")
        
    with rio.open(os.path.join(lisflood_simulation_dir, sample_file)) as src:
        height = src.height
        width = src.width
    return height, width

def generate_grid_sequences_light(lag: int, horizon: int, output_dir: str):
    """Generate grid-based sequences with upstream values only and save to HDF5 files"""
    logger.info("Loading upstream data and simulation files...")
    upstream_data = load_upstream_data()
    simulation_files = load_simulation_files()
    
    # Get grid dimensions
    height, width = get_grid_dimensions()
    num_cells = height * width
    logger.info(f"Grid dimensions: {height}x{width}, total cells: {num_cells}")
    
    # Split runs into train/validation/test
    train_runs = [run for run in simulation_files.keys() 
                 if run != TEST_SUBSET and run != VALIDATION_SUBSET]
    val_runs = [VALIDATION_SUBSET]
    test_runs = [TEST_SUBSET]
    
    splits = {
        'train': train_runs,
        'validation': val_runs,
        'test': test_runs
    }
    
    # Create output directory
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a metadata file to track all generated files
    metadata_file = os.path.join(output_dir, f"upstream_only_sequences_{date_time}_metadata.json")
    metadata = {
        "date_time": date_time,
        "lag": lag,
        "horizon": horizon,
        "grid_dimensions": {"height": height, "width": width},
        "files": {}
    }
    
    # Process each split
    for split_name, run_ids in splits.items():
        logger.info(f"Processing {split_name} split...")
        metadata["files"][split_name] = []
        
        # Create file for this split
        filename = f"upstream_seq_light_{date_time}_{split_name}_lag-{lag}_horizon-{horizon}.h5"
        file_path = os.path.join(output_dir, filename)
        
        # Number of features (only upstream values, no elevation or depth)
        num_features = 3  # Upstream1, Upstream2, Upstream3
        
        with h5py.File(file_path, 'w') as hf:
            # Create datasets
            X_dataset = hf.create_dataset('X', shape=(0, lag, num_features), 
                                  maxshape=(None, lag, num_features),
                                  chunks=(100, lag, num_features),
                                  compression='gzip')
            y_dataset = hf.create_dataset('y', shape=(0, height*width, horizon),
                                  maxshape=(None, height*width, horizon),
                                  chunks=(100, min(10000, height*width), horizon),
                                  compression='gzip')
                                  
            # Store metadata
            hf.attrs['split'] = split_name
            hf.attrs['grid_height'] = height
            hf.attrs['grid_width'] = width
            hf.attrs['lag'] = lag
            hf.attrs['horizon'] = horizon
            hf.attrs['features'] = ['Upstream1', 'Upstream2', 'Upstream3']
            
            seq_count = 0
            
            # Process each run in this split
            for run_id in run_ids:
                logger.info(f"Processing {split_name} - {run_id}")
                files = simulation_files[run_id]
                bc_data = upstream_data[run_id]
                
                # Calculate number of sequences
                total_timesteps = len(files)
                max_sequences = total_timesteps - lag - horizon + 1
                
                # Process in batches
                batch_size = 50  # Number of sequences to process at once
                
                for start_idx in range(0, max_sequences, batch_size):
                    end_idx = min(start_idx + batch_size, max_sequences)
                    X_batch = []
                    y_batch = []
                    
                    for seq_idx in range(start_idx, end_idx):
                        # For X: get upstream values for lag timesteps
                        X_seq = []
                        for t in range(seq_idx, seq_idx + lag):
                            bc_values = bc_data[bc_data['timestep'] == t].iloc[0]
                            upstream_values = [
                                bc_values['Upstream1'],
                                bc_values['Upstream2'],
                                bc_values['Upstream3']
                            ]
                            X_seq.append(upstream_values)
                        
                        # For y: get water depth for horizon timesteps
                        y_seq = np.zeros((height*width, horizon))
                        for h in range(horizon):
                            t_idx = seq_idx + lag + h
                            with rio.open(os.path.join(lisflood_simulation_dir, files[t_idx])) as src:
                                depth_data = src.read(1)
                                depth_data[depth_data < 0.3] = 0  # Filter small depths
                                y_seq[:, h] = depth_data.flatten()
                        
                        X_batch.append(X_seq)
                        y_batch.append(y_seq)
                    
                    # Convert to arrays
                    X_array = np.array(X_batch)
                    y_array = np.array(y_batch)
                    
                    # Add to HDF5
                    current_size = X_dataset.shape[0]
                    new_size = current_size + len(X_batch)
                    
                    X_dataset.resize(new_size, axis=0)
                    y_dataset.resize(new_size, axis=0)
                    
                    X_dataset[current_size:new_size] = X_array
                    y_dataset[current_size:new_size] = y_array
                    
                    seq_count += len(X_batch)
                    logger.info(f"{split_name} - {run_id}: Added {len(X_batch)} sequences, total: {seq_count}")
                    
                    # Clear memory
                    del X_batch, y_batch, X_array, y_array
                    gc.collect()
            
            logger.info(f"Completed {split_name} with {seq_count} sequences")
            
            # Add to metadata
            metadata["files"][split_name].append({
                "file_path": file_path,
                "sequences": seq_count,
                "run_ids": run_ids
            })
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Successfully saved all sequences to {output_dir}")
    logger.info(f"Metadata saved to {metadata_file}")
    
    return metadata_file

def load_grid_datasets_light(batch_size, epochs, lag, horizon):
    """Load grid datasets with only upstream features from the most recent HDF5 files"""
    # Find the most recent metadata file
    run_dir = os.path.join(PROJECT_ROOT, "carlisle-data", "grid_sequential_light")  # Make sure this matches the path in file_sequence_generator.py
    if not os.path.exists(run_dir):
        # Generate sequences if they don't exist
        logger.info("No existing upstream-only sequences found, generating new ones...")
        os.makedirs(run_dir, exist_ok=True)
        output_dir = os.path.join(run_dir, f"grid_sequences_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        metadata_file = generate_grid_sequences_light(lag, horizon, output_dir)
    else:
        # Look for metadata files
        metadata_files = []
        for subdir in os.listdir(run_dir):
            subdir_path = os.path.join(run_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith("_metadata.json") and "light" in file:
                        metadata_files.append(os.path.join(subdir_path, file))
                        
        if not metadata_files:
            # Generate sequences if no metadata found
            logger.info("No existing metadata found, generating new upstream-only sequences...")
            output_dir = os.path.join(run_dir, f"grid_sequences_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            metadata_file = generate_grid_sequences_light(lag, horizon, output_dir)
        else:
            # Use the latest metadata file based on timestamp in filename
            metadata_files.sort(reverse=True)
            metadata_file = metadata_files[0]
            logger.info(f"Using existing metadata file: {metadata_file}")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check if parameters match
    if metadata['lag'] != lag or metadata['horizon'] != horizon:
        logger.warning(f"Existing sequences have lag={metadata['lag']} and horizon={metadata['horizon']}, " +
                      f"but requested lag={lag} and horizon={horizon}")
        logger.info("Generating new upstream-only sequences with requested parameters...")
        output_dir = os.path.join(run_dir, f"grid_sequences_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
        metadata_file = generate_grid_sequences_light(lag, horizon, output_dir)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Load datasets
    train_file = metadata['files']['train'][0]['file_path']
    val_file = metadata['files']['validation'][0]['file_path']
    test_file = metadata['files']['test'][0]['file_path']
    
    # Get grid dimensions
    grid_height = metadata['grid_dimensions']['height']
    grid_width = metadata['grid_dimensions']['width']
    
    # Load training dataset
    train_h5 = h5py.File(train_file, 'r')
    X_train = train_h5['X']
    y_train = train_h5['y']
    train_size = X_train.shape[0]
    train_steps = train_size // batch_size
    
    # Load validation dataset
    val_h5 = h5py.File(val_file, 'r')
    X_val = val_h5['X']
    y_val = val_h5['y']
    val_size = X_val.shape[0]
    val_steps = val_size // batch_size
    
    # Load test dataset
    test_h5 = h5py.File(test_file, 'r')
    X_test = test_h5['X'][:]
    y_test = test_h5['y'][:]
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train[:], y_train[:]))
    train_dataset = train_dataset.batch(batch_size).repeat(epochs).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val[:], y_val[:]))
    val_dataset = val_dataset.batch(batch_size).repeat(epochs).prefetch(tf.data.AUTOTUNE)
    
    logger.info(f"Loaded upstream-only datasets: Train={train_size} samples, Val={val_size} samples, Test={X_test.shape[0]} samples")
    logger.info(f"Grid dimensions: {grid_height}x{grid_width}, Lag={lag}, Horizon={horizon}")
    logger.info(f"Train steps: {train_steps}, Val steps: {val_steps}")
    
    return train_dataset, val_dataset, X_test, y_test, grid_height, grid_width, train_steps, val_steps
