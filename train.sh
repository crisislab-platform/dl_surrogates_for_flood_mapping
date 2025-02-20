# Configuration for deep learning model training using parallel processing
# Format: GPU_ID LAG_WINDOW PREDICTION_HORIZON BATCH_SIZE LEARNING_RATE EPOCHS PATIENCE

# Run different combinations of parameters
parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python main.py train \
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  --patience {7} \
  ::: 0 \                    # GPU ID
  ::: 8 12 \                # Lag window sizes
  ::: 1 \                   # Prediction horizon
  ::: 128 256 \            # Batch sizes
  ::: 0.001 \              # Learning rates
  ::: 10 20 \             # Number of epochs
  ::: 5                   # Early stopping patience