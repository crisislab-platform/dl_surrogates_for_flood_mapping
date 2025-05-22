#!/bin/bash
# Sequential execution of SRR clustering with different parameter combinations
# This script runs the same parameter combinations as srr_clustering.sh but one at a time

echo "Starting sequential SRR clustering execution"
echo "============================================"

# Define parameter arrays
GPU_ID=0
SAMPLING_DISTS=(20 30 50 80 100)
N_CLUSTERS=(50 100 200)
RANDOM_STATE=42
N_INIT=10

# Count total combinations for progress tracking
TOTAL_COMBINATIONS=$((${#SAMPLING_DISTS[@]} * ${#N_CLUSTERS[@]}))
CURRENT=0

# Use nested loops instead of parallel
for SAMPLING_DIST in "${SAMPLING_DISTS[@]}"; do
  for N_CLUSTER in "${N_CLUSTERS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "[${CURRENT}/${TOTAL_COMBINATIONS}] Running with sampling_dist=${SAMPLING_DIST}, n_clusters=${N_CLUSTER}"
    
    # Execute one combination at a time
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3.11 main.py srr_cluster \
      --sampling_dist ${SAMPLING_DIST} \
      --n_clusters ${N_CLUSTER} \
      --random_state ${RANDOM_STATE} \
      --n_init ${N_INIT}
    
    echo "Completed: sampling_dist=${SAMPLING_DIST}, n_clusters=${N_CLUSTER}"
    echo "--------------------------------------------"
  done
done

echo "All SRR clustering combinations completed!"
echo "============================================"