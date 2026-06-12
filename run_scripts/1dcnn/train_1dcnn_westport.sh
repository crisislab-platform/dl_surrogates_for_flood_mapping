# parallel --line-buffer -j 1 \
#   systemd-run --scope -p MemoryMax=20G  \
#   /home/91/23016891/miniconda3/envs/westport-env/bin/python main.py train \
#   --model 1DCNN_V1 \
#   --lag {2} \
#   --horizon {3} \
#   --batch_size {4} \
#   --learning_rate {5} \
#   --epochs {6} \
#   --patience {7} \
#   --train_events  $(seq 1 8) 10 11 $(seq 13 17) $(seq 19 25) $(seq 27 37)  $(seq 39 72) $(seq 74 78)  80 81 83 $(seq 85 106)  \
#   --test_events 9 12 18 38 79 82 84 \
#   --validation_events 26 73 \
#   --tuning_mode \
#   --study_area westport \
#   --random_seed 42 \
#   ::: 0 \
#   ::: 8  \
#   ::: 1 \
#   ::: 8 \
#   ::: 0.01 0.001 \
#   ::: 10 \
#   ::: 5 

  parallel --line-buffer -j 1 \
  systemd-run --scope -p MemoryMax=20G  \
  /home/91/23016891/miniconda3/envs/westport-env/bin/python main.py train \
  --model 1DCNN_V1 \
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  --patience {7} \
  --train_events  $(seq 1 8) 10 11 $(seq 13 17) $(seq 19 37)  $(seq 39 78)  80 81 83 $(seq 85 106)  \
  --test_events 9 12 18 38 79 82 84 \
  --study_area westport \
  --do_profile \
  --random_seed {8} \
  ::: 0 \
  ::: 8  \
  ::: 1 \
  ::: 16 \
  ::: 0.01 \
  ::: 10 \
  ::: 5 \
  ::: 0 42 123 456 789