# Configuration for deep learning model training using parallel processing
PROJECT_DIR="/home/91/23016891/projects/carlisle"

parallel --line-buffer -j 6 --memfree 4G  CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
  --model USSR_1DCNN_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --n_clusters {6} \
  --sampling_dist {7} \
  --rl_group {8} \
  --input_time_len_h {9} \
  --usrr_conv_kernel {10} \
  --usrr_pool_kernel {11} \
  --patience {12} \
  --dropout {13} \
  --output_channel_size {14} \
  --fc_layer_size {15} \
  --save_model \
  ::: 0 \
  ::: 16 \
  ::: 2 \
  ::: 0.001 \
  ::: 50 \
  ::: 50 \
  ::: 50 \
  ::: $(seq 0 49) \
  ::: 8 \
  ::: 3 \
  ::: 3 \
  ::: 10 \
  ::: 0.2 \
  ::: 16 \
  ::: 64