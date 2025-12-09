from dotenv import load_dotenv
import os
load_dotenv()

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

study_area = os.getenv("STUDY_AREA", "carlisle").lower()

#Paths
PROJECT_ROOT = f"/home/91/23016891/projects/{study_area}"
OUTPUT_DIR = f"/data/{study_area}/out"
RUN_DIR = f"{OUTPUT_DIR}/runs"
PLOTS_OUTPUT_DIR = f"{OUTPUT_DIR}/plots"
SIMULATION_DATA_DIR = "/data/carlisle/DEM5m_2D"

if study_area == "carlisle":

    DEM_FILE = f"/{SIMULATION_DATA_DIR}/Carlisle_5m.asc"
    DATA_DIR = f"{PROJECT_ROOT}/data/{study_area}"
    SIMULATION_OUTPUT_DIR =  f"/data/{study_area}/simulation_output"
elif study_area == "westport":
    DATA_DIR = f"/data/{study_area}/data/flood-mapping-with-ml-main-DATA/DATA"
    DEM_FILE = f"{DATA_DIR}/Topo.tif"
    SIMULATION_DATA_DIR = f"/data/{study_area}/data"