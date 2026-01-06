# Set memory limit of 2GB for each process
parallel --line-buffer --memfree 4G CUDA_VISIBLE_DEVICES={1} python3.11 main.py train \
  --model HDL_FM_V1 \
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  ::: 0 \
  ::: 1 \
  ::: 1 \
  ::: 8 \
  ::: 0.001 \
  ::: 50 \
