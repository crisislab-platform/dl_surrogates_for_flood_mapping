parallel --line-buffer -j 1 \
  systemd-run --scope -p MemoryMax=30G  \
  /home/91/23016891/miniconda3/envs/westport-env/bin/python main.py train \
  --model HDL_FM_V1\
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  --patience {7} \
  --train_events  $(seq 1 8) 10 11 $(seq 13 17) $(seq 19 25) $(seq 27 37)  $(seq 39 72) $(seq 74 78)  80 81 83 $(seq 85 106)  \
  --test_events 9 12 18 38 79 82 84 \
  --validation_events 26 73 \
  --tuning_mode \
  --study_area westport \
  --indices_per_timestep {8} \
  --tile_resolution {9}\
  --sampling_dist {10} \
  --random_seed {11} \
  --sigma {12} \
  --patch_domain \
  ::: 0 \
  ::: 8 \
  ::: 1 \
  ::: 1 \
  ::: 0.00075 \
  ::: 15 \
  ::: 10 \
  ::: 1 \
  ::: 256 \
  ::: 128 \
  ::: 42  \
  ::: 2000

