
#findal script
parallel --line-buffer -j 3 \
  systemd-run --scope -p MemoryMax=20G  \
  /home/91/23016891/miniconda3/envs/westport-env/bin/python main.py train \
  --model 1DCNN_V1 \
  --lag {2} \
  --horizon {3} \
  --batch_size {4} \
  --learning_rate {5} \
  --epochs {6} \
  --train_events  $(seq 1 8) \
  --test_events 9 \
  --random_seed {7} \
  --study_area carlisle \
  --do_profile \
  ::: 0 \
  ::: 8 \
  ::: 1 \
  ::: 32 \
  ::: 0.01 \
  ::: 10 \
  ::: 0 42 123 456 789

#tuning script
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
#   --train_events  $(seq 1 8) \
#   --test_events 9 \
#   --tuning_mode \
#   --study_area carlisle \
#   ::: 0 \
#   ::: 8 \
#   ::: 1 \
#   ::: 4 8 16 32 \
#   ::: 0.001 0.01 \
#   ::: 10 20 50 \
#   ::: 5 10 \
