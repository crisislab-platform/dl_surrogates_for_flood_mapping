from modules.utils.path_util import DATA_DIR
import glob
import os

class DataLoader:
    def __init__(self, batch_size, sampling_dist, input_time_len_h, rl_group = None, validation = False , dev_mode = False):
        self.batch_size = batch_size
        self.input_time_len_h = input_time_len_h
        self.rl_group = rl_group
        self.validation = validation
        self.dev_mode = dev_mode
        self.rep_loc_cluster_meta_file = f"{DATA_DIR}/rls/run_meta_data.csv"
        self.rep_loc_file_path = f"{DATA_DIR}/rls/ss_{sampling_dist}.shp"
        self.dem_asc_file = f"{DATA_DIR}/Carlisle_5m.asc"
        self.simulation_dir = f"{DATA_DIR}/DEM5m_2D"
        self.possible_inun_file = f"{self.simulation_dir}/Run1-0175.wd"
        self.inundation_files = sorted(glob.glob(f"{self.simulation_dir}/*.wd"))

    def load_data(self):
        # Implement data loading logic here
        pass

    def preprocess_data(self):
        # Implement data preprocessing logic here
        pass

    def get_batch(self, batch_size):
        # Implement batch retrieval logic here
        pass