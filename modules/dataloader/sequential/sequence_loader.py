import h5py
import tensorflow as tf
import numpy as np
import logging
import time
import math
import json
import glob
from typing import Tuple, Dict, Optional, List, Iterator
from modules.utils.path_util import PROJECT_ROOT
import os

logger = logging.getLogger("GridSequenceLoader")

def find_sequence_directories(lag: int, horizon: int) -> List[str]:
    """Find all sequence directories that match lag and horizon"""
    grid_dir = os.path.join(PROJECT_ROOT, "carlisle-data", "grid_sequential")
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)
    
    # Look for directories that might contain our sequences
    dirs = [os.path.join(grid_dir, d) for d in os.listdir(grid_dir) 
            if os.path.isdir(os.path.join(grid_dir, d)) and d.startswith('grid_sequences_')]
    
    matching_dirs = []
    for d in dirs:
        # Check if there's a metadata file
        metadata_files = glob.glob(os.path.join(d, "*metadata.json"))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
                if metadata.get('lag') == lag and metadata.get('horizon') == horizon:
                    matching_dirs.append(d)
    
    # Sort by creation time (newest first)
    matching_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return matching_dirs

def find_sequence_files(split: str, lag: int, horizon: int) -> List[str]:
    """Find all sequence files for a specific split that match lag and horizon"""
    dirs = find_sequence_directories(lag, horizon)
    if not dirs:
        raise ValueError(f"No sequence directories found with lag={lag}, horizon={horizon}")
    
    # Get the most recent directory
    latest_dir = dirs[0]
    logger.info(f"Using sequence directory: {latest_dir}")
    
    # Find metadata file
    metadata_files = glob.glob(os.path.join(latest_dir, "*metadata.json"))
    if not metadata_files:
        # If no metadata, try to find matching files directly
        return glob.glob(os.path.join(latest_dir, f"*{split}*lag-{lag}_horizon-{horizon}.h5"))
    
    # Load metadata and get files for the requested split
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
        
    if split not in metadata.get('files', {}):
        raise ValueError(f"No {split} files found in metadata")
    
    # Extract file paths
    return [entry['file_path'] for entry in metadata['files'][split]]

def get_grid_dimensions_from_file(file_path: str) -> Tuple[int, int]:
    """Get grid dimensions from an H5 file"""
    with h5py.File(file_path, 'r') as hf:
        grid_height = hf.attrs.get('grid_height')
        grid_width = hf.attrs.get('grid_width')
    return grid_height, grid_width

class SequenceDataGenerator:
    """Generator that yields batches from multiple H5 files sequentially"""
    
    def __init__(self, file_paths: List[str], batch_size: int, epochs: int = 1):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.epochs = epochs
        self.total_samples = 0
        self.steps_per_epoch = 0
        
        # Calculate total samples across all files
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as hf:
                self.total_samples += hf['X'].shape[0]
        
        self.steps_per_epoch = math.ceil(self.total_samples / batch_size)
    
    def get_steps(self) -> int:
        """Get steps per epoch"""
        return self.steps_per_epoch
    
    def get_dataset(self) -> tf.data.Dataset:
        """Create a TF dataset from the generator"""
        
        # First get shapes from the first file
        with h5py.File(self.file_paths[0], 'r') as hf:
            X_shape = hf['X'].shape[1:]  # Skip the batch dimension
            y_shape = hf['y'].shape[1:]
        
        # Create generator function
        def generator_fn():
            for _ in range(self.epochs):
                for file_path in self.file_paths:
                    with h5py.File(file_path, 'r') as hf:
                        total_samples = hf['X'].shape[0]
                        chunk_size = self.batch_size # Process in chunks
                        
                        for i in range(0, total_samples, chunk_size):
                            end_idx = min(i + chunk_size, total_samples)
                            chunk_X = hf['X'][i:end_idx]
                            chunk_y = hf['y'][i:end_idx]
                            
                            # Yield individual samples
                            for j in range(chunk_X.shape[0]):
                                yield chunk_X[j], chunk_y[j]
        
        # Create the dataset
        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_signature=(
                tf.TensorSpec(shape=X_shape, dtype=tf.float32),
                tf.TensorSpec(shape=y_shape, dtype=tf.float32)
            )
        )
        
        # Apply batching
        return dataset.prefetch(tf.data.AUTOTUNE)

def load_multi_file_grid_datasets(batch_size: int, epochs: int, lag: int, horizon: int) -> Tuple:
    """Load grid-based datasets from multiple files"""
    try:
        # Find files for each split
        train_files = find_sequence_files('train', lag, horizon)
        val_files = find_sequence_files('validation', lag, horizon)
        test_files = find_sequence_files('test', lag, horizon)
        
        logger.info(f"Found {len(train_files)} train files, {len(val_files)} validation files, {len(test_files)} test files")
        
        # Get grid dimensions (should be the same in all files)
        grid_height, grid_width = get_grid_dimensions_from_file(train_files[0])
        logger.info(f"Grid dimensions: {grid_height}x{grid_width}")
        
        # Create generators for train and validation
        train_generator = SequenceDataGenerator(train_files, batch_size, epochs)
        val_generator = SequenceDataGenerator(val_files, batch_size)
        
        train_data = train_generator.get_dataset()
        val_data = val_generator.get_dataset()
        
        train_steps = train_generator.get_steps()
        val_steps = val_generator.get_steps()
        
        # For test data, we'll load just one file at a time when needed
        # Here we're just getting the first test file as a placeholder
        with h5py.File(test_files[0], 'r') as hf:
            x_test = hf['X'][:]
            y_test = hf['y'][:]
        
        logger.info(f"Train steps: {train_steps}, Validation steps: {val_steps}")
        
        # Return only the 8 expected values, omitting test_files
        return train_data, val_data, x_test, y_test, grid_height, grid_width, train_steps, val_steps
        
    except Exception as e:
        logger.error(f"Error loading multi-file grid datasets: {e}")
        raise

# Update the original load_grid_datasets to use the new multi-file approach
def load_grid_datasets(batch_size: int, epochs: int, lag: int, horizon: int) -> Tuple:
    """Load grid-based datasets for training, validation and test"""
    try:
        return load_multi_file_grid_datasets(batch_size, epochs, lag, horizon)
        
    except Exception as e:
        logger.error(f"Error loading grid datasets: {e}")
        raise
