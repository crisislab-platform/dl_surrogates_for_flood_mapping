from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Reshape, TimeDistributed, Conv2DTranspose
import tensorflow as tf
from lib.commons import ModelConfig
from modules.dataloader.sequential.sequence_loader import load_grid_datasets
from modules.dataloader.sequential.sequence_loader_light import load_grid_datasets_light
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GridConvLSTM_Trainer")

def create_grid_data_generators(lag, horizon, batch_size, epochs=5):
    """Create data generators for grid-based training"""
    train_dataset, val_dataset, x_test, y_test, grid_height, grid_width, train_steps, val_steps = load_grid_datasets(
        batch_size, epochs, lag, horizon
    )
    
    logger.info(f"Loaded dataset, calculating steps per epoch")
    logger.info(f"Grid dimensions: {grid_height}x{grid_width}")
    logger.info(f"Train steps: {train_steps}, Validation steps: {val_steps}")
    
    return train_dataset, val_dataset, x_test, y_test, train_steps, val_steps, grid_height, grid_width

def transform_data_for_convlstm(dataset, grid_height, grid_width, lag, batch_size, horizon=None):
    """Transform data to work with ConvLSTM by reshaping from flattened grid to 2D grid"""
    def reshape_batch(x, y):
        # x is [batch, lag, num_features]  # Modified to handle upstream-only features
        # y is [batch, num_cells, horizon]
        
        # Reshape y from [batch, num_cells, horizon] to [batch, height, width, horizon]
        y_reshaped = tf.reshape(y, [batch_size, grid_height, grid_width, horizon])
        
        # Create a grid where each cell has the same upstream values
        # Expand upstream values to fill the entire grid
        x_expanded = tf.tile(
            tf.reshape(x, [batch_size, lag, 1, 1, 3]),  # 3 upstream features
            [1, 1, grid_height, grid_width, 1]
        )
        
        return x_expanded, y_reshaped
    
    # Apply the transformation
    return dataset.map(lambda x, y: reshape_batch(x, y))

def transform_data_for_convlstm_upstream_only(dataset, grid_height, grid_width, lag, batch_size, horizon=None):
    """Transform upstream-only data without expanding to grid format for input"""
    def reshape_batch(x, y):
        # x is [batch, lag, num_features=3]  # Only upstream features
        # y is [batch, num_cells, horizon]
        
        # Reshape y from [batch, num_cells, horizon] to [batch, height, width, horizon]
        y_reshaped = tf.reshape(y, [batch_size, grid_height, grid_width, horizon])
        
        # No need to expand x to a grid, keep as is
        return x, y_reshaped
    
    # Apply the transformation
    return dataset.map(lambda x, y: reshape_batch(x, y))

def create_convlstm_model(config: ModelConfig, grid_height: int, grid_width: int) -> tf.keras.Model:
    """Create ConvLSTM model that maintains spatial structure of grid cells"""
    try:
        # Input shape: (lag, height, width, features) where features are only upstream values (3)
        inputs = Input(shape=(config.lag, grid_height, grid_width, 3))  # 3 features: upstream1, upstream2, upstream3
        
        # ConvLSTM layers - maintain spatial awareness
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', 
                      return_sequences=True, activation='tanh',
                      recurrent_dropout=config.dropout_rate)(inputs)
        x = BatchNormalization()(x)
        
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', 
                      return_sequences=False, activation='tanh',
                      recurrent_dropout=config.dropout_rate)(x)
        x = BatchNormalization()(x)
        
        # Convolutional layers instead of dense layers
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        
        # Output layer: predict horizon steps for each grid cell
        outputs = Conv2D(filters=config.horizon, kernel_size=(1, 1), padding='same', activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        model.summary(print_fn=logger.info)
        return model
        
    except Exception as e:
        logger.error(f"Grid ConvLSTM model creation failed: {e}")
        raise

def create_convlstm_model_upstream_only(config: ModelConfig, grid_height: int, grid_width: int) -> tf.keras.Model:
    """Create ConvLSTM model for upstream-only features"""
    try:
        # Input shape: (lag, height, width, features) where features are only upstream values (3)
        inputs = Input(shape=(config.lag, grid_height, grid_width, 3))  # 3 features: upstream1, upstream2, upstream3
        
        # ConvLSTM layers - maintain spatial awareness
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', 
                      return_sequences=True, activation='tanh',
                      recurrent_dropout=config.dropout_rate)(inputs)
        x = BatchNormalization()(x)
        
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', 
                      return_sequences=False, activation='tanh',
                      recurrent_dropout=config.dropout_rate)(x)
        x = BatchNormalization()(x)
        
        # Convolutional layers instead of dense layers
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        
        # Output layer: predict horizon steps for each grid cell
        outputs = Conv2D(filters=config.horizon, kernel_size=(1, 1), padding='same', activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        model.summary(print_fn=logger.info)
        return model
        
    except Exception as e:
        logger.error(f"Grid ConvLSTM model creation failed: {e}")
        raise

def get_subsampled_grid_dimensions(height, width, max_cells):
    """Calculate target dimensions that keep total cells under max_cells while maintaining aspect ratio"""
    current_cells = height * width
    if current_cells <= max_cells:
        return height, width
    
    scale = np.sqrt(max_cells / current_cells)
    target_height = int(height * scale)
    target_width = int(width * scale)
    return target_height, target_width

def spatial_downsampling(dataset, target_height, target_width, original_height, original_width, lag, batch_size, horizon):
    """Downsample spatial dimensions of the dataset"""
    def downsample_batch(x, y):
        # Reshape x to include spatial dimensions
        x = tf.reshape(x, [batch_size, lag, original_height, original_width, 5])
        y = tf.reshape(y, [batch_size, original_height, original_width, horizon])
        
        # Resize using bilinear interpolation
        x = tf.image.resize(tf.reshape(x, [-1, original_height, original_width, 5]), 
                          [target_height, target_width])
        x = tf.reshape(x, [batch_size, lag, target_height, target_width, 5])
        
        y = tf.image.resize(y, [target_height, target_width])
        return x, y
    
    return dataset.map(downsample_batch)

def get_grid_convlstm_model(lag, horizon, batch_size, learning_rate, max_grid_cells=70000):
    """Create and setup grid-based ConvLSTM model with memory optimizations for upstream-only features"""
    config = ModelConfig(
        lag=lag,
        horizon=horizon,
        batch_size=batch_size,
        learning_rate=learning_rate, 
        model_name="ConvLSTM_UpstreamOnly",
        dropout_rate=0.2,
        mixed_precision=True
    )
    
    # Get data generators and grid info
    train_dataset, val_dataset, x_test, y_test, grid_height, grid_width, training_steps, validation_steps = load_grid_datasets(
        config.batch_size, epochs=5, lag=config.lag, horizon=config.horizon
    )
    
    # Check if we need to downsample the grid to save memory
    target_height, target_width = get_subsampled_grid_dimensions(
        grid_height, grid_width, max_cells=max_grid_cells
    )
    
    if target_height != grid_height or target_width != grid_width:
        logger.info(f"Downsampling grid from {grid_height}x{grid_width} to {target_height}x{target_width}")
        # Transform datasets to lower resolution
        train_dataset_reshaped = spatial_downsampling(
            train_dataset, target_height, target_width, grid_height, grid_width, 
            lag, batch_size, horizon
        )
        val_dataset_reshaped = spatial_downsampling(
            val_dataset, target_height, target_width, grid_height, grid_width, 
            lag, batch_size, horizon
        )
        # Update grid dimensions
        grid_height, grid_width = target_height, target_width
    else:
        # Transform datasets to work with ConvLSTM
        train_dataset_reshaped = transform_data_for_convlstm(train_dataset, grid_height, grid_width, lag, batch_size, horizon)
        val_dataset_reshaped = transform_data_for_convlstm(val_dataset, grid_height, grid_width, lag, batch_size, horizon)
    
    # Create model with memory-optimized architecture
    model = create_memory_optimized_convlstm_model(config, grid_height, grid_width)
    
    return model, train_dataset_reshaped, val_dataset_reshaped, training_steps, validation_steps, config

def create_memory_optimized_convlstm_model(config: ModelConfig, grid_height: int, grid_width: int) -> tf.keras.Model:
    """Create a memory-optimized ConvLSTM model for upstream-only features"""
    try:
            # Enable mixed precision
            if config.mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                # Input shape: (lag, height, width, features) where features are only upstream values (3)
                inputs = Input(shape=(config.lag, grid_height, grid_width, 3))
                
                # Apply strided convolution to reduce spatial dimensions immediately
                x = TimeDistributed(Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same'))(inputs)
                
                # First ConvLSTM layer with reduced filters
                x = ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', 
                            return_sequences=True, activation='tanh',
                            recurrent_dropout=config.dropout_rate)(x)
                x = BatchNormalization()(x)
                
                # Second ConvLSTM layer
                x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', 
                            return_sequences=False, activation='tanh',
                            recurrent_dropout=config.dropout_rate)(x)
                x = BatchNormalization()(x)
                
                # Transposed convolution to restore original dimensions
                x = Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
                
                # Output layer
                outputs = Conv2D(filters=config.horizon, kernel_size=(1, 1), padding='same', activation='linear')(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                    loss='mse',
                    metrics=['mae', 'mse']
                )
                
                model.summary(print_fn=logger.info)
                return model
        
    except Exception as e:
        logger.error(f"Grid ConvLSTM model creation failed: {e}")
        raise

def create_memory_optimized_convlstm_model_upstream_only(config: ModelConfig, grid_height: int, grid_width: int) -> tf.keras.Model:
    """Create a simplified model for upstream-only features with minimal reshaping"""
    try:
        # Enable mixed precision
        if config.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Simple Input: (lag, 3 upstream features)
            inputs = Input(shape=(config.lag, 3), name="input_layer")
            
            # Basic time-series processing layers
            x = tf.keras.layers.Flatten()(inputs)  # Flatten to (batch, lag*3)
            
            # Simple dense layers
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(config.dropout_rate)(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(config.dropout_rate)(x)
            
            # Output layer - directly predict flattened grid cells
            output_size = grid_height * grid_width * config.horizon
            outputs = tf.keras.layers.Dense(output_size)(x)
            outputs = tf.keras.layers.Reshape((grid_height, grid_width, config.horizon))(outputs)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            model.summary(print_fn=logger.info)
            return model
    
    except Exception as e:
        logger.error(f"Upstream-only model creation failed: {e}")
        raise

def prepare_dataset_for_training(dataset, grid_height, grid_width, target_height, target_width, batch_size, horizon):
    """Prepare dataset for training by reshaping both X and Y to match expected dimensions"""
    def transform_batch(x, y):
        # For target data: reshape from [batch, cells, horizon] to [batch, target_height, target_width, horizon]
        # First to original dimensions, then resize
        y_orig = tf.reshape(y, [batch_size, grid_height, grid_width, horizon])
        y_resized = tf.image.resize(y_orig, [target_height, target_width])
        
        # X is already in correct shape for upstream only: [batch, lag, 3]
        return x, y_resized
    
    return dataset.map(transform_batch)

def get_grid_convlstm_model_upstream_only(lag, horizon, batch_size, learning_rate, max_grid_cells=70000):
    """Create and setup simplified upstream-only model with proper data transformation"""
    config = ModelConfig(
        lag=lag,
        horizon=horizon,
        batch_size=batch_size,
        learning_rate=learning_rate, 
        model_name="ConvLSTM_UpstreamOnly",
        dropout_rate=0.2,
        mixed_precision=True
    )
    
    # Get data generators with the light loader (upstream-only features)
    train_dataset, val_dataset, x_test, y_test, grid_height, grid_width, training_steps, validation_steps = load_grid_datasets_light(
        config.batch_size, epochs=5, lag=config.lag, horizon=config.horizon
    )
    
    logger.info(f"Loaded upstream-only dataset with grid dimensions: {grid_height}x{grid_width}")
    
    # Use smaller grid dimensions if needed
    if grid_height * grid_width > max_grid_cells:
        scale = np.sqrt(max_grid_cells / (grid_height * grid_width))
        target_height = max(1, int(grid_height * scale))
        target_width = max(1, int(grid_width * scale))
        logger.info(f"Using smaller grid: {grid_height}x{grid_width} → {target_height}x{target_width}")
    else:
        target_height, target_width = grid_height, grid_width
    
    # Transform datasets to ensure X and Y have compatible dimensions    
    train_dataset = prepare_dataset_for_training(
        train_dataset, grid_height, grid_width, target_height, target_width, 
        batch_size, horizon
    )
    
    val_dataset = prepare_dataset_for_training(
        val_dataset, grid_height, grid_width, target_height, target_width,
        batch_size, horizon
    )
        
    # Create the simplified model
    model = create_memory_optimized_convlstm_model_upstream_only(config, target_height, target_width)
    
    # Return everything needed for training
    return model, train_dataset, val_dataset, training_steps, validation_steps, config
