# Configuration for deep learning model training using parallel processing
PROJECT_DIR="/home/91/23016891/projects/carlisle"

parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
  --model USSR_1DCNN_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --patience {6} \
  --n_clusters {7} \
  --sampling_dist {8} \
  --rl_group {9} \
  --input_time_len_h {10} \
  --tuning_mode true \
  --fold {11} \
  ::: 0 \
  ::: 32 \
  ::: 2 \
  ::: 0.001 \
  ::: 50 \
  ::: 10 \
  ::: 50 \
  ::: 20 50 100 200 \
  ::: 1 \
  ::: 10 \
  ::: $(seq 1 8)