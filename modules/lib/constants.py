# Model names
LSTM_V1 = "LSTM_V1"
CNN1D_V1 = "1DCNN_V1"
PICNN1D_V1 = "PICNN1D_V1"
LSTM_SRR_V1 = "LSTM_SRR_V1"
USRR_1DCNN_V1 = "USSR_1DCNN_V1"
USSR_1DCNN_V2 = "USSR_1DCNN_V2"
USRR_UNET_V1 = "USSR_UNET_V1"
USRR_CNN1D_COMBINED = "USSR_CNN1D_COMBINED"
SRR_LSTM_COMBINED = "SRR_LSTM_COMBINED"
POD_BNN_V1 = "POD_BNN_V1"
TCN_V1 = "TCN_V1"
HDL_FM_V1 = "HDL_FM_V1"
USRR_LSTM = "USRR_LSTM_V1"

# Metrics
RMSE = "RMSE"
MRMSE = "mRMSE"
RMSE_T = "RMSE_t"
RMSE_S = "RMSE_s"
MRMSE_T = "mRMSE_t"
MRMSE_S = "mRMSE_s"
HITRATE = "Hitrate"
CSI = "CSI"
F2SCORE = "F2Score"
F3SCORE = "F3Score"
INFERENCE_TIMES = "Latency"
INFERENCE_MEMORY_USAGE = "Memory"
FLOPS = "FLOPs"
PARAMS = "Parameters"

#Paths
STUDY_AREA = "westport"  # Change to "westport" for Westport study area
PROJECT_ROOT = f"/home/91/23016891/projects/{STUDY_AREA}"
DATA_DIR = f"/data/{STUDY_AREA}"
RUN_DIR = f"{DATA_DIR}/runs"
OUTPUT_DIR = f"{PROJECT_ROOT}/out"
PLOTS_OUTPUT_DIR = f"{OUTPUT_DIR}/plots"
SIMULATION_DATA_DIR = f"{DATA_DIR}/simulation" if STUDY_AREA == "carlisle" else f"{DATA_DIR}/data/flood-mapping-with-ml-main-DATA/DATA"
DEM_FILE = f"{DATA_DIR}/simulation/Carlisle_5m.asc" if STUDY_AREA == "carlisle" else f"{SIMULATION_DATA_DIR}/Topo.tif"
FLOOD_MAPS_DIR = f"{DATA_DIR}/data/flood_maps" if STUDY_AREA == "westport" else SIMULATION_DATA_DIR
BC_DATA_DIR = f"{SIMULATION_DATA_DIR}/boundary_conditions" if STUDY_AREA == "westport" else SIMULATION_DATA_DIR
MODEL_CHECKPOINT_DIR = f"{RUN_DIR}/model_checkpoints"

