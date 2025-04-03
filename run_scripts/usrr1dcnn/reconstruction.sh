parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python3.11 main.py srr_reconstruction \
  --sampling_dist {2} \
  --n_clusters {3} \
  ::: 0 \
  ::: 20 \
  ::: 100 
