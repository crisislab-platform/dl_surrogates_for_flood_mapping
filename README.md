# Deep Learning Based Flood Inundation Modeling

This project implements deep learning models to predict flood inundation patterns using LISFLOOD-FP simulation data. The system is trained on historical flood data from Carlisle, UK to predict future flood extents.

## Prerequisites

- LISFLOOD-FP simulator
- Python 3.8+
- CUDA-capable GPU
- GNU Parallel
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

```
carlisle/
├── carlisle-data/         # Simulation data and parameters
├── models/                # Trained model checkpoints
├── src/                   # Source code
├── train.sh              # Training script
├── simulation_data_generator.sh  # Data generation script
└── README.md
```

## Usage

### 1. Generate Simulation Dataset

First, generate the flood simulation dataset using LISFLOOD-FP:

```bash
cd carlisle-data
chmod +x simulation_data_generator.sh
./simulation_data_generator.sh
```

The script will:
- Run 9 different flood scenarios
- Generate output files for each simulation
- Create a log file (simulation.log) tracking progress

### 2. Train the Model

The training script uses GNU Parallel to experiment with different hyperparameters:

```bash
./train.sh
```

Training parameters:
- Lag windows: 8, 12 (historical time steps)
- Prediction horizon: 1 (future time steps)
- Batch sizes: 128, 256
- Learning rate: 0.001
- Epochs: 10, 20
- Early stopping patience: 5

## Model Parameters

### Lag Window
The number of historical time steps used to predict future flood extent.
- Smaller values (e.g., 8): Faster training, but may miss longer-term patterns
- Larger values (e.g., 12): Better for capturing longer-term dependencies

### Prediction Horizon
Number of time steps to predict into the future (currently set to 1).

### Batch Size
Number of samples processed before model update:
- 128: Better for limited GPU memory
- 256: Faster training on powerful GPUs

### Learning Rate
Controls how much to adjust the model in response to errors (0.001 is a stable choice).

### Early Stopping
Training stops if no improvement is seen for 5 epochs to prevent overfitting.

## Monitoring

- Training progress is logged to stdout
- Model checkpoints are saved in the `models/` directory
- Simulation logs are stored in `carlisle-data/simulation.log`

## Running the ConvLSTM Model

The ConvLSTM model uses spatial-temporal features by preserving the grid structure and using convolutional LSTM cells. This model is more memory-efficient for large grid sizes as it uses parameter sharing.

### Using the ConvLSTM model

Train the ConvLSTM model:
```
