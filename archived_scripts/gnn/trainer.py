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
    
        
    
    