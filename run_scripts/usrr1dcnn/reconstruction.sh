parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 main.py train \
  --model USSR_CNN1D_COMBINED \
  --batch_size {2} \
  --sampling_dist {3} \
  --n_clusters {4} \
  ::: 0 \
  ::: 16 \
  ::: 50 \
  ::: 50 