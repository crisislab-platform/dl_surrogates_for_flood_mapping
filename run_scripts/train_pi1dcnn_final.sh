# Set memory limit of 2GB for each process
parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 main.py train \
  --model PICNN1D_V1 \
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  --patience {7} \
  ::: 0 \
  ::: 8 \
  ::: 1 \
  ::: 32 \
  ::: 0.001 \
  ::: 100 \
  ::: 10 \
