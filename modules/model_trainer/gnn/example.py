import numpy as np
import logging
import tensorflow as tf
from trainer import GNNTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Example usage of the TensorFlow GNN model
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data for demonstration
    # In a real scenario, this would be your actual data
    n_samples = 100
    n_nodes = 20
    node_features = 16
    
    # Generate random features (each sample is a graph with n_nodes)
    X = np.random.randn(n_samples, n_nodes, node_features)
    
    # Generate random labels
    y = np.random.randn(n_samples)
    
    # Split data into training and validation sets
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logging.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    
    # Initialize and train the GNN model
    trainer = GNNTrainer(
        input_dim=node_features, 
        hidden_dim=64, 
        output_dim=1,
        learning_rate=0.001,
        num_layers=3,
        dropout=0.3
    )
    
    # Train the model
    history = trainer.train(
        X_train, 
        y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val)
    )
    
    # Make predictions
    predictions = trainer.predict(X_val)
    logging.info(f"Predictions shape: {predictions.shape}")
    
    # Save the model
    trainer.save_model('/home/91/23016891/projects/carlisle/models/gnn_model')

if __name__ == "__main__":
    main()
