#!/bin/bash

# Set parameters
DEVICE=0
LAG=8
HORIZON=1
# Define arrays for batch sizes and learning rates
BATCH_SIZES=(32 64)
LEARNING_RATES=(0.01 0.001)
EPOCHS=50
PATIENCE=10
TUNING_MODE=true
# Define arrays for folds (no spaces around = sign)
FOLDS=(1 2 3 4 5 6 7 8)
# If you want to process folds in groups:
GROUP1=(1 2 3)
GROUP2=(4 5 6)
GROUP3=(7 8)

echo "Starting sequential training of Pi1DCNN with multiple hyperparameters"
echo "=================================================================="

# Loop through each learning rate
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
  # Loop through each batch size
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "Configuration: Learning Rate=${LEARNING_RATE}, Batch Size=${BATCH_SIZE}"
    echo "------------------------------------------------------------------"
    
    # Comment out single fold processing approach
    # for FOLD in "${FOLDS[@]}"; do
    #   echo "Running fold ${FOLD} with LR=${LEARNING_RATE}, BS=${BATCH_SIZE}..."
    #   
    #   CUDA_VISIBLE_DEVICES=${DEVICE} python3.11 main.py train \
    #     --model PICNN1D_V1 \
    #     --lag ${LAG} \
    #     --horizon ${HORIZON} \
    #     --batch_size ${BATCH_SIZE} \
    #     --learning_rate ${LEARNING_RATE} \
    #     --epochs ${EPOCHS} \
    #     --patience ${PATIENCE} \
    #     --fold ${FOLD} \
    #     --tuning_mode ${TUNING_MODE}
    #   
    #   echo "Fold ${FOLD} completed."
    #   echo "--------------------------------------------------"
    # done
    
    # Process folds in groups
    echo "Processing group 1 folds (${GROUP1[*]}) in parallel"
    parallel --line-buffer CUDA_VISIBLE_DEVICES=${DEVICE} python3.11 main.py train \
      --model PICNN1D_V1 \
      --lag ${LAG} \
      --horizon ${HORIZON} \
      --batch_size ${BATCH_SIZE} \
      --learning_rate ${LEARNING_RATE} \
      --epochs ${EPOCHS} \
      --patience ${PATIENCE} \
      --fold {} \
      --tuning_mode ${TUNING_MODE} \
      ::: "${GROUP1[@]}"
    
    echo "Group 1 completed."

    echo "Processing group 2 folds (${GROUP2[*]}) in parallel"
    parallel --line-buffer CUDA_VISIBLE_DEVICES=${DEVICE} python3.11 main.py train \
      --model PICNN1D_V1 \
      --lag ${LAG} \
      --horizon ${HORIZON} \
      --batch_size ${BATCH_SIZE} \
      --learning_rate ${LEARNING_RATE} \
      --epochs ${EPOCHS} \
      --patience ${PATIENCE} \
      --fold {} \
      --tuning_mode ${TUNING_MODE} \
      ::: "${GROUP2[@]}"
    
    echo "Group 2 completed."

    echo "Processing group 3 folds (${GROUP3[*]}) in parallel"
    parallel --line-buffer CUDA_VISIBLE_DEVICES=${DEVICE} python3.11 main.py train \
      --model PICNN1D_V1 \
      --lag ${LAG} \
      --horizon ${HORIZON} \
      --batch_size ${BATCH_SIZE} \
      --learning_rate ${LEARNING_RATE} \
      --epochs ${EPOCHS} \
      --patience ${PATIENCE} \
      --fold {} \
      --tuning_mode ${TUNING_MODE} \
      ::: "${GROUP3[@]}"
    
    echo "Group 3 completed."
    
    echo "Completed all fold groups for LR=${LEARNING_RATE}, BS=${BATCH_SIZE}"
    echo "=================================================================="
  done
done

echo "All configurations and folds completed!"
