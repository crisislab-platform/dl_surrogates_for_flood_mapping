import torch.nn as nn
from modules.models.model_wrapper import ModelWrapper, ModelConfig
import os

class GNNModelWrapper(ModelWrapper):
    """
    Graph Neural Network model using TensorFlow GNN
    """
    def __init__(self, config):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_layers = config.num_layers
    
    def init_model(self):
        pass

    def train(self):
        train_dataset =  None
        val_dataset = None
        history = {
            'train_loss': [],
            'val_loss': []
        }
        for epoch in range(self.config.epochs):
            pass
        return history
    
    def predict(self):
        pass

    def validate_model(self):
        pass

    
class GNNModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_layers = []
        
    def forward(self, x, edge_index, batch):
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

        