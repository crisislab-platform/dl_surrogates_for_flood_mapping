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
  --train_events  $(seq 1 7) $(seq 10 28)  $(seq 30 35) $(seq 37 64) $(seq 66 71) $(seq 73 82) $(seq 85 100) $(seq 102 106)  \
  --test_events 9 29 36 65 72 83 84 101 \
  --validation_events 8 \
  --tuning_mode \
  --study_area westport \
  ::: 0 \
  ::: 8 \
  ::: 1 \
  ::: 8 \
  ::: 0.001 \
  ::: 50 \
  ::: 10 \