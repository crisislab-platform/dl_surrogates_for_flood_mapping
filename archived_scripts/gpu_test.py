import tensorflow as tf
import time
import numpy as np
import logging
from tensorflow.keras.models import Sequential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPU_TEST")

def test_gpu():
    """Test GPU availability and performance"""
    
    # Check CUDA and CUDNN versions
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    logger.info(f"CUDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Num GPUs Available: {len(gpus)}")
    
    if len(gpus) == 0:
        logger.warning("No GPU found!")
        return
        
    # Configure memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Test 1: Basic tensor operations
    logger.info("Test 1: Basic tensor operations")
    with tf.device('/GPU:0'):
        a = tf.random.normal([10000, 1000])
        b = tf.random.normal([1000, 2000])
        start_time = time.time()
        c = tf.matmul(a, b)
        logger.info(f"Matrix multiplication time: {time.time() - start_time:.2f} seconds")
    
    # Test 2: Memory allocation
    logger.info("Test 2: Memory allocation test")
    try:
        with tf.device('/GPU:0'):
            # Allocate 2GB tensor
            large_tensor = tf.random.normal([1024, 1024, 256])
            logger.info(f"Successfully allocated large tensor: {large_tensor.shape}")
    except Exception as e:
        logger.error(f"Memory allocation failed: {e}")
    
    # Test 3: Small model training
    logger.info("Test 3: Training test")
    with tf.device('/GPU:0'):
        model = Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        x = tf.random.normal([10000, 100])
        y = tf.random.normal([10000, 1])
        
        model.compile(optimizer='adam', loss='mse')
        start_time = time.time()
        model.fit(x, y, epochs=5, batch_size=32, verbose=0)
        logger.info(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # Simple matrix multiplication test
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        
        start_time = time.time()
        c = tf.matmul(a, b)
        duration = time.time() - start_time
        
        logger.info(f"GPU Matrix multiplication time: {duration:.4f} seconds")
        logger.info(f"Operation device: {c.device}")

if __name__ == "__main__":
    test_gpu()