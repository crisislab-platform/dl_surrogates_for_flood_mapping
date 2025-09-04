from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.models.srr_lstm.srr.srr_reconstruction import SDRReconstructor
from modules.lib.constants import SRR_LSTM_COMBINED, RUN_DIR
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SRR_LSTM_Model")

class SRRLSTMModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = SRR_LSTM_COMBINED
        self.reconstructor = SDRReconstructor()
        
    def init_model(self):
        return True
    
    def train(self, run_dir, tuning_mode=False):
        # Return metrics of sub-models
        lstm_training_metrics = self.reconstructor.lstm_metrics
        
        total_train_time = lstm_training_metrics['train_time']
        memory_usage = lstm_training_metrics['training_memory_usage']
  
        reduction_file = os.path.join(RUN_DIR, "SRR_LSTM_REDUCTION", "reduction_metrics.csv")
        if os.path.exists(reduction_file):
            reduction_metrics = pd.read_csv(reduction_file)
            #sort by run_id and get the last row
            reduction_metrics = reduction_metrics.sort_values(by='run_id').iloc[-1]
            reduction_time = reduction_metrics['reduction_time']
            logger.info(f"Reduction time: {reduction_time:.2f} seconds")
            max_reduction_cpu_memory = reduction_metrics['max_cpu_memory']
            total_cpu_time = reduction_metrics['total_cpu_time']
            logger.info(f"Max CPU memory used during reduction: {max_reduction_cpu_memory:.2f} MB")
            logger.info(f"Total CPU time used during reduction: {total_cpu_time:.2f} seconds")
            
            total_train_time += reduction_time
            if  memory_usage['max_cpu_memory'] < max_reduction_cpu_memory:
                memory_usage['max_cpu_memory'] = max_reduction_cpu_memory
                
            memory_usage['total_cpu_time'] = memory_usage['total_cpu_time'] + total_cpu_time
        else:
            logger.warning(f"Reduction metrics file {reduction_file} does not exist.")

        history = {
            "train_time": total_train_time,
            "memory": memory_usage,
            "trainable_params": lstm_training_metrics['trainable_params'],
            "total_neurons": lstm_training_metrics['total_neurons'],
        }
        return history, total_train_time, None

    def test_model(self):
        metrics = SDRReconstructor().reconstruct()
        return metrics
       
        