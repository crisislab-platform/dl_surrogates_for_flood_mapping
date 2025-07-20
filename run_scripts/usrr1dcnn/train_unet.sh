PROJECT_DIR="/home/91/23016891/projects/carlisle"

# 1DCNN_V1 training with all parameters, batch size here is the number of batches per timestep(map)
parallel --line-buffer --memfree 4G CUDA_VISIBLE_DEVICES={1} python3.11 "${PROJECT_DIR}/main.py" train \
  --model USSR_UNET_V1 \
  --batch_size {2} \
  --lag {3} \
  --learning_rate {4} \
  --epochs {5} \
  --patience {6} \
  --sampling_dist {7} \
  --save_model \
  ::: 0 \
  ::: 16 \
  ::: 2 \
  ::: 0.001 \
  ::: 10 \
  ::: 10 \
  ::: 50
