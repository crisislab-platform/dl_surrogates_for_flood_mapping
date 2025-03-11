import tensorflow as tf
from tensorflow import keras
import tensorflow_gnn as tfgnn
from tensorflow_gnn.keras.layers import GraphUpdate, NodeSetUpdate, Pool
import os

class GNN(keras.Model):
    """
    Graph Neural Network model using TensorFlow GNN
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Define GNN layers
        self.gnn_layers = []
        for _ in range(num_layers):
            self.gnn_layers.append(self._create_graph_update_layer(hidden_dim))
        
        # Output layers
        self.global_pool = Pool(tfgnn.CONTEXT, "mean")
        self.dropout = keras.layers.Dropout(dropout)
        self.output_layer = keras.layers.Dense(output_dim)
    
    def _create_graph_update_layer(self, hidden_dim):
        """Create a graph update layer for message passing"""
        # Node update function (MLP)
        node_set_update = NodeSetUpdate(
            {"nodes": tfgnn.keras.layers.NextStateFromConcat(
                keras.Sequential([
                    keras.layers.Dense(hidden_dim, activation="relu"),
                    keras.layers.Dense(hidden_dim)
                ])
            )},
            edge_set_inputs={
                "edges": tfgnn.keras.layers.EdgeSetUpdate(
                    keras.Sequential([
                        keras.layers.Dense(hidden_dim, activation="relu")
                    ]),
                    "sum"
                )
            }
        )
        
        return GraphUpdate(node_sets={"nodes": node_set_update})
    
    def call(self, graph, training=False):
        """
        Forward pass for the GNN model
        
        Args:
            graph: TensorFlow GNN graph tensor
            training: Whether in training mode
            
        Returns:
            tf.Tensor: Output predictions
        """
        # Apply GNN layers
        for layer in self.gnn_layers:
            graph = layer(graph)
        
        # Global pooling
        x = self.global_pool(graph)
        
        # Apply dropout during training
        if training:
            x = self.dropout(x)
        
        # Final prediction
        return self.output_layer(x)
    
    def save(self, path):
        """Save the model to the specified path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.save_weights(path)
    
    def load(self, path):
        """Load the model from the specified path"""
        self.load_weights(path)
