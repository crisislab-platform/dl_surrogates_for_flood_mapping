import numpy as np
import tensorflow as tf

def spatial_downsampling(dataset, target_height=100, target_width=150, original_height=611, original_width=951, 
                          lag=24, batch_size=8, horizon=12):
    """Downsample the spatial dimensions of the grid dataset"""
    
    def downsample_batch(x, y):
        # x is [batch, lag, height, width, features]
        # y is [batch, height, width, horizon]
        
        # Reshape x to combine batch and lag dimensions for easier downsampling
        x_reshaped = tf.reshape(x, [-1, original_height, original_width, 5])
        
        # Downsample x using resize
        x_down = tf.image.resize(x_reshaped, [target_height, target_width])
        
        # Reshape back to original format with new dimensions
        x_final = tf.reshape(x_down, [batch_size, lag, target_height, target_width, 5])
        
        # Downsample y 
        y_reshaped = tf.reshape(y, [batch_size, original_height, original_width, horizon])
        y_down = tf.image.resize(y_reshaped, [target_height, target_width])
        
        return x_final, y_down
    
    return dataset.map(downsample_batch)

def get_subsampled_grid_dimensions(original_height=611, original_width=951, max_cells=70000):
    """Calculate appropriate grid dimensions to stay under memory limits"""
    original_cells = original_height * original_width
    if original_cells <= max_cells:
        return original_height, original_width
    
    # Keep aspect ratio while reducing total cells
    ratio = original_width / original_height
    new_height = int(np.sqrt(max_cells / ratio))
    new_width = int(new_height * ratio)
    
    return new_height, new_width
