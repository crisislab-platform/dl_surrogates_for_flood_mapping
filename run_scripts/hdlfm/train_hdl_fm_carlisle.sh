parallel --line-buffer -j 2 \
  systemd-run --scope -p MemoryMax=20G  \
  /home/91/23016891/miniconda3/envs/westport-env/bin/python main.py train \
  --model HDL_FM_V1 \
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  --patience {7} \
  --train_events  $(seq 1 7) \
  --test_events 9 \
  --validation_events 8 \
  --indices_per_timestep {8} \
  --tile_resolution {9} \
  --sampling_dist {10} \
  --tuning_mode \
  --study_area carlisle \
  --random_seed {11} \
  --patch_domain \
  --sigma {12} \
  ::: 0 \
  ::: 8 \
  ::: 1 \
  ::: 8 \
  ::: 0.00075 \
  ::: 50 \
  ::: 15 \
  ::: 1 \
  ::: 256 \
  ::: 128 \
  ::: 42  \
  ::: 2000


  # parallel --line-buffer -j 2 \
  # systemd-run --scope -p MemoryMax=20G  \
  # /home/91/23016891/miniconda3/envs/westport-env/bin/python main.py train \
  # --model HDL_FM_V1 \
  # --lag {2} \
  # --horizon {3} \
  # --batch_size {4} \
  # --learning_rate {5} \
  # --epochs {6} \
  # --patience {7} \
  # --train_events  $(seq 1 8) \
  # --test_events 9 \
  # --indices_per_timestep {8} \
  # --tile_resolution {9} \
  # --sampling_dist {10} \
  # --study_area carlisle \
  # --random_seed {11} \
  # --do_profile \
  # --sigma {12} \
  # ::: 0 \
  # ::: 8 \
  # ::: 1 \
  # ::: 32 \
  # ::: 0.0001 \
  # ::: 35 \
  # ::: 10 \
  # ::: 3 \
  # ::: 256 \
  # ::: 128 \
  # ::: 0 42 123 456 789 \
  # ::: 2000