import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import time
from dataloader import FloodDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from preprocessor import get_num_features

# Configure logging and CUDA
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSTM_model_training")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Constants
LAG = 8
FORECAST_HORIZON = 1
BATCH_SIZE = 5000
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
TRAIN_SUBSET_IDENTIFIER = 'train'
VAL_SUBSET_IDENTIFIER = 'test'
PATIENCE = 2
EPOCHS = 10

class FloodLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, FORECAST_HORIZON)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    num_features = get_num_features()
    model = FloodLSTM(num_features).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Setup data loaders
    train_dataset = FloodDataGenerator(subset=TRAIN_SUBSET_IDENTIFIER)
    val_dataset = FloodDataGenerator(subset=VAL_SUBSET_IDENTIFIER)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        
        logger.info(f'Epoch {epoch+1}/{EPOCHS}')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info('Early stopping triggered')
                break

if __name__ == "__main__":
    main()