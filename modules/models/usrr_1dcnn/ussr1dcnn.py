from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.models.usrr_1dcnn.unet import UNetModelWrapper
from modules.models.usrr_1dcnn.cnn1d import CNN1DModelWrapper
from modules.models.usrr_1dcnn.reduction.rl_culster_finder import RLClusterFinder
from modules.lib.constants import USRR_CNN1D_COMBINED, RUN_DIR
import logging
import os
from modules.models.usrr_1dcnn.reconstruction.reconstruction import ReconstructionModule
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USSR_1D_CNN_Model")

class USSR1DCNNModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = USRR_CNN1D_COMBINED
        self.cnn1d_models = "tbd"
        self.sampling_dist =  config.args['sampling_dist']
        self.n_clusters = config.args['n_clusters']
        run_dir = os.path.join(config.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        self.reco_module = ReconstructionModule(self.sampling_dist, self.n_clusters, run_dir, config.batch_size)
        
    def init_model(self):
        return True
    
    def train(self, run_dir, tuning_mode=False):
        # Return metrics of sub-models
        reduction_file = os.path.join(RUN_DIR, "USRR_1DCNN_REDUCTION", "reduction_metrics.csv")
        if os.path.exists(reduction_file):
            reduction_metrics = pd.read_csv(reduction_file)
            #sort by run_id and get the last row
            reduction_metrics = reduction_metrics.sort_values(by='run_id').iloc[-1]
            reduction_time = reduction_metrics['reduction_time']
            logger.info(f"Reduction time: {reduction_time:.2f} seconds")
            max_reduction_cpu_memory = reduction_metrics['max_cpu_memory']
            max_reduction_gpu_memory = reduction_metrics['max_cuda_memory']
            total_cpu_time = reduction_metrics['total_cpu_time']
            total_gpu_time = reduction_metrics['total_gpu_time']
            logger.info(f"Max CPU memory used during reduction: {max_reduction_cpu_memory:.2f} MB")
            logger.info(f"Total CPU time used during reduction: {total_cpu_time:.2f} seconds")
        else:
            logger.warning(f"Reduction metrics file {reduction_file} does not exist.")
            
        
        cnn_metrics = self.reco_module.cnn_train_history
        unet_metrics = self.reco_module.unet_train_history
        total_train_time = cnn_metrics['train_time'] + unet_metrics['train_time'] + reduction_time
        
        memory_usage = {
            'max_cpu_memory': max(cnn_metrics['memory']['max_cpu_memory'], unet_metrics['memory']['max_cpu_memory'], max_reduction_cpu_memory),
            'total_cpu_time': cnn_metrics['memory']['total_cpu_time'] + unet_metrics['memory']['total_cpu_time'] + total_cpu_time,
            'total_gpu_time': cnn_metrics['memory']['total_gpu_time'] + unet_metrics['memory']['total_gpu_time'] + total_gpu_time,  # Assuming no GPU usage in this model
            'max_cuda_memory': max(cnn_metrics['memory']['max_cuda_memory'], unet_metrics['memory']['max_cuda_memory'], max_reduction_gpu_memory),
            'total_cpu_memory': 0, 
            'total_gpu_memory': 0  
        }
        
        trainable_params = cnn_metrics['trainable_params'] + unet_metrics['trainable_params']
        total_neurons = cnn_metrics['total_neurons'] + unet_metrics['total_neurons']
        
        history = {
            "train_time": total_train_time,
            "memory": memory_usage,
            "trainable_params": trainable_params,
            "total_neurons": total_neurons,
        }
        return history, total_train_time, None

    def test_model(self):
        return self.reco_module.reconstruct()
        

