#!/bin/bash
# Configuration for spatial reduction representative clustering with parallel processing

# Run SRR clustering with different parameter combinations
# Parameters: {1}=GPU_ID, {2}=sampling_dist, {3}=n_clusters, {4}=random_state, {5}=n_init for clustering

parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 main.py srr_cluster \
  --sampling_dist {2} \
  --n_clusters {3} \
  --random_state {4} \
  --n_init {5} \
  ::: 0 \
  ::: 50  \
  ::: 100 \
  ::: 42 \
  ::: 10