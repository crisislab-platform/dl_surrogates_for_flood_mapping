#!/bin/bash

# Configuration for parallel database initialization
# Format: WINDOW_LENGTH

echo "Starting parallel database initialization at $(date)"

# Run different window lengths in parallel
parallel --line-buffer python main.py db_init \
  --window_length {1} \
  ::: 0 # Different window sizes to try

echo "Completed all database initializations at $(date)"

# You can also initialize with full window
echo "Initializing with full window..."
python main.py db_init
