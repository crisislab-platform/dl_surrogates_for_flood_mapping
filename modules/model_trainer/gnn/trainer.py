import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import logging
import time

from .gnn import GNN
from .data_processor import GraphDataProcessor

class GNNTrainer:
    """
    Trainer class for Graph Neural Network models using TensorFlow
    """
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001, num_layers=3, dropout=0.5):
        """
        Initialize the GNN trainer
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
            learning_rate (float): Learning rate
            num_layers (int): Number of GNN layers
            dropout (float): Dropout rate
        """
        self.model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        # For regression tasks
        self.loss_fn = keras.losses.MeanSquaredError()
        self.data_processor = GraphDataProcessor(k_neighbors=10)
        
        # Metrics
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
        
        logging.info(f"TensorFlow GNN model initialized with {num_layers} layers")
    
    @tf.function
    def _train_step(self, graph):
        """Single training step"""
        # Get label from graph context
        y_true = graph.context['label']
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(graph, training=True)
            # Calculate loss
            loss = self.loss_fn(y_true, y_pred)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss_metric.update_state(loss)
        
        return loss
    
    @tf.function
    def _val_step(self, graph):
        """Single validation step"""
        # Get label from graph context
        y_true = graph.context['label']
        
        # Forward pass
        y_pred = self.model(graph, training=False)
        # Calculate loss
        loss = self.loss_fn(y_true, y_pred)
        
        # Update metrics
        self.val_loss_metric.update_state(loss)
        
        return loss
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        """
        Train the GNN model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_data (tuple, optional): (X_val, y_val) for validation
            
        Returns:
            dict: Training history
        """
        # Prepare training data
        train_dataset = self.data_processor.prepare_data(X_train, y_train, batch_size=batch_size)
        
        # Prepare validation data if available
        val_dataset = None
        if validation_data:
            X_val, y_val = validation_data
            val_dataset = self.data_processor.prepare_data(X_val, y_val, batch_size=batch_size)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Reset metrics at the start of each epoch
            self.train_loss_metric.reset_states()
            if val_dataset:
                self.val_loss_metric.reset_states()
            
            # Training phase
            for batch in train_dataset:
                self._train_step(batch)
            
            # Validation phase
            if val_dataset:
                for batch in val_dataset:
                    self._val_step(batch)
                
                history['val_loss'].append(self.val_loss_metric.result().numpy())
                logging.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Time: {time.time() - start_time:.2f}s - "
                    f"Loss: {self.train_loss_metric.result():.4f} - "
                    f"Val Loss: {self.val_loss_metric.result():.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Time: {time.time() - start_time:.2f}s - "
                    f"Loss: {self.train_loss_metric.result():.4f}"
                )
            
            history['train_loss'].append(self.train_loss_metric.result().numpy())
        
        return history
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the provided data
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Labels
            
        Returns:
            float: Evaluation loss
        """
        # Prepare evaluation data
        eval_dataset = self.data_processor.prepare_data(X, y, batch_size=32, shuffle=False)
        
        # Reset metrics
        self.val_loss_metric.reset_states()
        
        # Evaluation loop
        for batch in eval_dataset:
            self._val_step(batch)
        
        return self.val_loss_metric.result().numpy()
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        # Prepare data for prediction (without labels)
        pred_dataset = self.data_processor.prepare_data(X, batch_size=32, shuffle=False)
        
        predictions = []
        
        # Prediction loop
        for batch in pred_dataset:
            pred = self.model(batch, training=False)
            predictions.append(pred.numpy())
        
        return np.vstack(predictions) if predictions else np.array([])
    
    def save_model(self, path):
        """
        Save the model to the specified path
        
        Args:
            path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the model from the specified path
        
        Args:
            path (str): Path to load the model from
        """
        self.model = keras.models.load_model(path)
        logging.info(f"Model loaded from {path}")
