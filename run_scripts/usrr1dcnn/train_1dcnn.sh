# Configuration for deep learning model training using parallel processing
PROJECT_DIR="/home/91/23016891/projects/carlisle"

parallel --line-buffer -j 4 CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
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
  --tuning_mode true \
  --fold {12} \
  --patience 10 \
  --save_model \
  ::: 0 \
  ::: 32 \
  ::: 2 \
  ::: 0.01 \
  ::: 50 \
  ::: 200 \
  ::: 30 \
  ::: 0  \
  ::: 6 \
  ::: 4 \
  ::: 3 \
  ::: $(seq 2 8)



# #
# PROJECT_DIR="/home/91/23016891/projects/carlisle"

# parallel --line-buffer -j 4 CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
#   --model USSR_1DCNN_V1 \
#   --batch_size {2} \
#   --lag {3} \
#   --learning_rate {4} \
#   --epochs {5} \
#   --n_clusters {6} \
#   --sampling_dist {7} \
#   --rl_group {8} \
#   --input_time_len_h {9} \
#   --usrr_conv_kernel {10} \
#   --usrr_pool_kernel {11} \
#   ::: 0 \
#   ::: 32 \
#   ::: 2 \
#   ::: 0.01 \
#   ::: 10 \
#   ::: 200 \
#   ::: 30 \
#   :::  $(seq 0 199) \
#   ::: 10 \
#   ::: 4 \
#   ::: 3 \