# Configuration for LSTM model training using parallel processing
PROJECT_DIR="/home/91/23016891/projects/carlisle"

parallel --line-buffer -j 1 --memfree 4G  CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
  --model USSR_1DCNN_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --n_clusters {6} \
  --sampling_dist {7} \
  --rl_group {8} \
  --input_time_len_h {9} \
  --num_layers {10} \
  --hidden_size {14} \
  --dropout {16} \
  --fc_layer_size {17} \
  --save_model \
  ::: 0 \
  ::: 64 \
  ::: 2 \
  ::: 0.005 \
  ::: 50 \
  ::: 50 \
  ::: 50 \
  ::: $(seq 0 49) \
  ::: 12 \
  ::: 2 \
  ::: 64 \
  ::: 15 \
  ::: 0.2 \
  ::: 32

  
