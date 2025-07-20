import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, steps, features, outputs):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * steps, 32)
        self.bn_fc1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, outputs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv1d expects input shape: [batch_size, channels, sequence_length]
        # But our data comes in as: [batch_size, sequence_length, features]
        x = x.transpose(1, 2)
        
        # First convolutional block with batch norm
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second convolutional block with batch norm
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        
        # First fully connected block with batch norm and dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)

        # Second fully connected block with batch norm and dropout
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
     
        # Third fully connected block with batch norm
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = self.relu(x)
        
        # Output layer
        x = self.fc4(x)
        return x
            