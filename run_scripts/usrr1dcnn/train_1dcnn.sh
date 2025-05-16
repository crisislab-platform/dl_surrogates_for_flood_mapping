# Configuration for deep learning model training using parallel processing
PROJECT_DIR="/home/91/23016891/projects/carlisle"
n_clusters=50

# Run parallel jobs with different sampling_dist and matching rl_group ranges
# For sampling_dist=20, use rl_group 0-20

parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
  --model USSR_1DCNN_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --patience {6} \
  --n_clusters $n_clusters \
  --sampling_dist 20 \
  --rl_group {7} \
  --input_time_len_h {8} \
  --tuning_mode {9} \
  --fold {10} \
  ::: 0 \
  ::: 32 \
  ::: 2 \
  ::: 0.001 \
  ::: 50 \
  ::: 5 \
  ::: 1 \
  ::: 10 \
  ::: true \
  ::: $(seq 1 8)