PROJECT_DIR="/home/91/23016891/projects/carlisle"

# Set CUDA memory management environment variables with 6GB limit
# 1DCNN_V1 training with all parameters, batch size here is the number of batches per timestep(map)
# ulimit -v sets virtual memory limit (6GB = 6291456 KB), --memfree 4G ensures 4GB free system memory
parallel --line-buffer --memfree 6G  "CUDA_VISIBLE_DEVICES={1} python3.11 ${PROJECT_DIR}/main.py" train \
  --model USSR_UNET_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --patience {6} \
  --sampling_dist {7} \
  --save_model \
  ::: 0 \
  ::: 19 \
  ::: 2 \
  ::: 0.001 \
  ::: 10 \
  ::: 10 \
  ::: 50
