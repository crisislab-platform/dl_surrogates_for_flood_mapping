parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 main.py train \
  --model USSR_CNN1D_COMBINED \
  --n_clusters {2} \
  --sampling_dist {3}  \
  --run_id {4} \
  ::: 0 \
  ::: 50 \
  ::: 20 \
  ::: 20250411_095632
