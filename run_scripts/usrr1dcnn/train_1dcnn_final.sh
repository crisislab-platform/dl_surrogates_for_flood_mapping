PROJECT_DIR="/home/91/23016891/projects/carlisle"

parallel --line-buffer -j 6 CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
  --model USSR_1DCNN_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --n_clusters {6} \
  --sampling_dist {7} \Manu
  --rl_group {8} \
  --input_time_len_h {9} \
  --usrr_conv_kernel {10} \
  --usrr_pool_kernel {11} \
  --save_model \
  ::: 0 \
  ::: 32 \
  ::: 2 \
  ::: 0.01 \
  ::: 10 \
  ::: 200 \
  ::: 30 \
  :::  $(seq 0 199) \
  ::: 10 \
  ::: 4 \
  ::: 3

