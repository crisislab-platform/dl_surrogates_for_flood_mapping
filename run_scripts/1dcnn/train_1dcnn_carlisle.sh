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
  --train_events  $(seq 3 9)\
  --test_events 1 \
  --validation_events 2 \
  --tuning_mode \
  --study_area carlisle \
  ::: 0 \
  ::: 8 \
  ::: 1 \
  ::: 32 \
  ::: 0.001 \
  ::: 50 \
  ::: 10 \