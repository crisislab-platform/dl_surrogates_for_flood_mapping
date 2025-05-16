from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.models.usrr_1dcnn.unet import UNetModelWrapper
from modules.models.usrr_1dcnn.cnn1d import CNN1DModelWrapper
from modules.models.usrr_1dcnn.spatial_reduction_module.rl_culster_finder import RLClusterFinder
from modules.models.usrr_1dcnn.spatial_reduction_module.reconstruction import validate_reconstruction
from modules.lib.constants import USRR_CNN1D_COMBINED
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USSR_1D_CNN_Model")


class USSR1DCNNModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = USRR_CNN1D_COMBINED
        self.cnn1d_models = "tbd"
        self.sampling_dist =  config.args['sampling_dist']
        self.n_clusters = config.args['n_clusters']
        
    def init_model(self):
        return True
    
    def train(self, run_dir):
        return {}, 0, None

    def test_model(self):
        metrics = validate_reconstruction(self.sampling_dist, self.n_clusters, self.config.run_dir)
        return metrics
        

