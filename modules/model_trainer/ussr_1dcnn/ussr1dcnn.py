from modules.model_trainer.model import Model, ModelConfig
from modules.model_trainer.ussr_1dcnn.unet import UNetModelWrapper
from modules.model_trainer.ussr_1dcnn.cnn1d import CNN1DModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("USSR_1D_CNN_Model")

class USSR1DCNNModelConfig(ModelConfig):
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    model_name: str
    dropout_rate: float = 0.2
    epochs: int = 10
    mixed_precision: bool = False

class USSR1DCNNModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = "USSR_1D_CNN_V1"
        self.sampling_dist  =  config.args.get('sampling_dist', 300)
        self.unet = UNetModelWrapper(config)
        # self.cnn1d = CNN1DModel()
        
    def create_dataset(self):
        pass

    def init_model(self) -> bool:
        try:
            status = self.unet.init_model()
            logger.info(f"USSR 1D CNN model initialized")
            # self.cnn1d.init_model()
            return status
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False
    
    def train(self, run_dir: str):
        history, train_time = self.unet.train(run_dir)
        # cnn1d_model_file = self.cnn1d.train(run_dir) 
        # return unet_model_file, cnn1d_model_file #Check'
        return history, train_time
    
    def predict(self, run_id = None, model_file = None):
        return "Not implemented"

