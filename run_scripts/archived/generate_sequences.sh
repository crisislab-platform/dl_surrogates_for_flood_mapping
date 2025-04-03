parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python main.py generate_grid_sequences_light \
  --lag {2} \
  --horizon {3} \
  ::: 0 \
  ::: 8 \
  ::: 1