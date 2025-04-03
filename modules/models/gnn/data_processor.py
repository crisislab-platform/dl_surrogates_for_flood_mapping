import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from sklearn.neighbors import NearestNeighbors

class GraphDataProcessor:
    """
    Process data into graph format for TensorFlow GNN models
    """
    def __init__(self, k_neighbors=10):
        """
        Initialize the data processor
        
        Args:
            k_neighbors (int): Number of neighbors to connect in the graph
        """
        self.k_neighbors = k_neighbors
    
    def _create_graph_tensor(self, features, label=None):
        """
        Create a graph tensor from feature matrix using k-nearest neighbors
        
        Args:
            features (np.ndarray): Feature matrix of shape (num_nodes, feature_dim)
            label (float, optional): Label for the graph
            
        Returns:
            tfgnn.GraphTensor: Graph tensor for the input
        """
        num_nodes = features.shape[0]
        
        # Find k-nearest neighbors
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto')
        knn.fit(features)
        _, indices = knn.kneighbors(features)
        
        # Create edges (skip the first column as it's the node itself)
        source_nodes = np.repeat(np.arange(num_nodes), self.k_neighbors)
        target_nodes = indices[:, 1:].flatten()
        
        # Create bidirectional edges for undirected graph
        source_nodes_bidirectional = np.concatenate([source_nodes, target_nodes])
        target_nodes_bidirectional = np.concatenate([target_nodes, source_nodes])
        
        # Create graph schema
        schema = tfgnn.GraphSchema()
        schema.node_sets["nodes"] = tfgnn.NodeSetSchema(features=tfgnn.Schema.FeatureSpec(shape=(features.shape[1],), dtype=tf.float32))
        schema.edge_sets["edges"] = tfgnn.EdgeSetSchema(
            source="nodes",
            target="nodes",
            features={}
        )
        if label is not None:
            schema.context = tfgnn.ContextSchema(
                features={"label": tfgnn.Schema.FeatureSpec(shape=(), dtype=tf.float32)}
            )
        
        # Create graph tensor
        graph_tensor_dict = {
            "nodes": {
                "features": tf.convert_to_tensor(features, dtype=tf.float32),
                tfgnn.SIZES_NAME: tf.constant([num_nodes], dtype=tf.int32)
            },
            "edges": {
                tfgnn.SOURCE_NAME: tf.convert_to_tensor(source_nodes_bidirectional, dtype=tf.int32),
                tfgnn.TARGET_NAME: tf.convert_to_tensor(target_nodes_bidirectional, dtype=tf.int32),
                tfgnn.SIZES_NAME: tf.constant([len(source_nodes_bidirectional)], dtype=tf.int32)
            },
            tfgnn.CONTEXT: {}
        }
        
        if label is not None:
            graph_tensor_dict[tfgnn.CONTEXT]["label"] = tf.convert_to_tensor([label], dtype=tf.float32)
            graph_tensor_dict[tfgnn.CONTEXT][tfgnn.SIZES_NAME] = tf.constant([1], dtype=tf.int32)
        
        return tfgnn.GraphTensor.from_pieces(
            node_sets={"nodes": tfgnn.NodeSet.from_fields(**graph_tensor_dict["nodes"])},
            edge_sets={"edges": tfgnn.EdgeSet.from_fields(**graph_tensor_dict["edges"])},
            context=tfgnn.Context.from_fields(**graph_tensor_dict[tfgnn.CONTEXT])
        )
    
    def prepare_data(self, X, y=None, batch_size=32, shuffle=True):
        """
        Prepare data for GNN training or inference
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray, optional): Labels (if available)
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            tf.data.Dataset: TensorFlow dataset for graph data
        """
        graph_tensors = []
        
        for i in range(len(X)):
            # Create graph tensor from features
            label = None if y is None else y[i]
            graph_tensor = self._create_graph_tensor(X[i], label)
            graph_tensors.append(graph_tensor)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(graph_tensors)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset
