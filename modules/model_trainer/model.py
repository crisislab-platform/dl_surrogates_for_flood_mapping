from tensorflow.keras.models import Sequential
from dataclasses import dataclass
import tensorflow as tf
import os
import json
import gc
import logging
import time

logger = logging.getLogger("Model")

# Unified Model configuration
@dataclass
class ModelConfig:
    """Unified configuration class for model parameters and training"""
    model_name: str
    lag: int
    horizon: int
    batch_size: int
    learning_rate: float
    epochs: int = 10
    patience: int = 2
    dropout_rate: float = 0.2
    mixed_precision: bool = False
    memory_limit: int = None

class Model:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.train_dataset = None
        self.val_dataset = None
        self.x_test = None
        self.y_test = None
        self.grid_height = None
        self.grid_width = None
        self.train_steps = None
        self.val_steps = None
        self.model = None
        
    def init_model(self) -> bool:
        pass
    
    def create_dataset(self):
        pass
    
    def train(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        try:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.patience,
                    restore_best_weights=True,
                    mode='min'
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(run_dir, 'logs'),
                    update_freq='epoch'
                )
            ]

            # Save model configuration
            with open(os.path.join(run_dir, 'model_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=2)

            logger.info("Training model started")
            start_time = time.time()    
            
            history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=self.config.epochs,
                callbacks=callbacks,
                steps_per_epoch=self.train_steps,
                validation_steps=self.val_steps,
                verbose=1
            )
            
            end_time = time.time()
            train_time = end_time - start_time
            logger.info("Training completed in %.2f minutes" % ((end_time - start_time)/60))
            return history, train_time
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            gc.collect()
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
                
    def predict(self, run_id: str = None, model_file: str = None):
        """Predict using the model. Implementation should be provided by subclasses."""
        pass
