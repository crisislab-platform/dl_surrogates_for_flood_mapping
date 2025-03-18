# Configuration for deep learning model training using parallel processing

# LSTM_V1 training with all parameters
parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 main.py train \
  --model USSR_1D_CNN_V1 \
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  --patience {7} \
  --sampling_dist {8} \
  ::: 0 \
  ::: 8 \
  ::: 1 \
  ::: 32 \
  ::: 0.01 \
  ::: 100 \
  ::: 10 \
  ::: 300