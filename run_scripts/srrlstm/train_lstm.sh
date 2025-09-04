# Configuration for deep learning model training using parallel processing
PROJECT_DIR="/home/91/23016891/projects/carlisle"

parallel --line-buffer -j 6 --memfree 4G  CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
  --model LSTM_SRR_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --rl_id {6} \
  --input_time_len_h {7} \
  --patience {8} \
  --dropout {9} \
  --save_model \
  ::: 0 \
  ::: 32 \
  ::: 2 \
  ::: 0.005 \
  ::: 100 \
  ::: $(seq 1 50) \
  ::: 12 \
  ::: 10 \
  ::: 0.2 




  
