parallel --line-buffer --memfree 4G CUDA_VISIBLE_DEVICES={1} python3.11 main.py train \
  --model SRR_LSTM_COMBINED \
  --batch_size {2} \
  ::: 0 \
  ::: 1