from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.models.usrr_1dcnn.unet import UNetModelWrapper
from modules.models.usrr_1dcnn.cnn1d import CNN1DModelWrapper
from modules.models.usrr_1dcnn.reduction.rl_culster_finder import RLClusterFinder
from modules.models.usrr_1dcnn.reconstruction.reconstruction import reconstruct_and_test
from modules.lib.constants import SRR_LSTM_COMBINED
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SRR_LSTM_Model")

class SRRLSTMModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = SRR_LSTM_COMBINED
        self.lstm_models = None
        self.sampling_dist =  config.args['sampling_dist']
        self.n_clusters = config.args['n_clusters']
        
    def init_model(self):
        return True
    
    def train(self, run_dir, tuning_mode=False):
        return {}, 0, None

    def test_model(self):
        metrics = reconstruct_and_test(self.sampling_dist, self.n_clusters, self.config.run_dir, self.config.batch_size)
        return metrics
        