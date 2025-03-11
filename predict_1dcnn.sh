parallel --line-buffer CUDA_VISIBLE_DEVICES={1} python main.py predict \
    --model {2} \
    --run_id {3} \
    ::: 0 \
    ::: 1DCNN_V1 \
    ::: 20250304_112026