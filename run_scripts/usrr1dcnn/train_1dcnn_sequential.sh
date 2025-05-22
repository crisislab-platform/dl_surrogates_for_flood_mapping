# Configuration for deep learning model training using sequential processing
PROJECT_DIR="/home/91/23016891/projects/carlisle"

# Find optimal sampling distance and number of clusters based on the performance of the model
# Fixed parameters
CUDA_DEVICE=0
BATCH_SIZE=32
LAG=2
LEARNING_RATE=0.001
EPOCHS=50
PATIENCE=10
INPUT_TIME_LEN_H=10
RL_GROUP=1

# Parameters to iterate
N_CLUSTERS=(50 100 200)
SAMPLING_DISTS=(20 30 50 80 100)
FOLDS=(5 6 7 8)

# Process each sampling distance and fold sequentially, but n_clusters in parallel
for SAMPLING_DIST in "${SAMPLING_DISTS[@]}"; do
        echo "====================================="
        echo "Starting batch: sampling_dist=$SAMPLING_DIST"
        echo "====================================="
        
        parallel --jobs 12 --line-buffer \
        "echo 'Starting job: n_clusters={1} sampling_dist=${SAMPLING_DIST} fold={2}' && \
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3.11 ${PROJECT_DIR}/main.py train \
        --model USSR_1DCNN_V1 \
        --batch_size ${BATCH_SIZE} \
        --lag ${LAG} \
        --learning_rate ${LEARNING_RATE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --n_clusters {1} \
        --sampling_dist ${SAMPLING_DIST} \
        --rl_group ${RL_GROUP} \
        --input_time_len_h ${INPUT_TIME_LEN_H} \
        --tuning_mode true \
        --fold {2} && \
        echo 'Completed job: n_clusters={1} sampling_dist=${SAMPLING_DIST} fold={2}'" ::: "${N_CLUSTERS[@]}" ::: "${FOLDS[@]}"
        
        echo "Batch completed: sampling_dist=$SAMPLING_DIST"
        # Add a small delay between sets of parallel jobs
        sleep 2
done

