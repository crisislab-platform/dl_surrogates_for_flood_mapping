#!/bin/bash
# Training script for USSR1DCNN components

# Exit on error
set -e

# Configuration
PROJECT_DIR="/home/91/23016891/projects/carlisle"

echo "=== USRR1DCNN Training Script Started at $(date) ==="
# Step 1: Run the SRR clustering script
echo "Step 1: Running SRR clustering script to generate clustering results..." 
# bash "${PROJECT_DIR}/run_scripts/usrr1dcnn/srr_clustering.sh"
if [ $? -ne 0 ]; then
    echo "Error: SRR clustering script failed!" 
fi
echo "SRR clustering completed successfully." 

# Step 2: Train the UNet model with various configurations
echo "Step 2: Training UNet model in parallel..."
bash "${PROJECT_DIR}/run_scripts/usrr1dcnn/train_unet.sh" 

# Step 3: Train the 1DCNN model
echo "Step 3: Training 1DCNN model in parallel..."s
# bash "${PROJECT_DIR}/run_scripts/usrr1dcnn/train_1dcnn.sh" &

# Wait for all background processes to complete
wait

echo "UNet and 1DCNN training completed successfully."
echo "=== USRR1DCNN Training Script Completed at $(date) ==="
