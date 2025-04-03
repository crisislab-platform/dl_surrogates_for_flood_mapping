from modules.models.model_wrapper import ModelWrapper, ModelConfig
from modules.utils.run_util import check_device
from modules.dataloader.sequential_1dcnn.cnn_datamanager import CNNDataManager
from modules.lib.constants import USRR_1DCNN_V1
import numpy as np
import torch
import torch.nn as nn
import logging
import time
import os


model_name = USRR_1DCNN_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNN1DModel")

class CNN1DModelWrapper(ModelWrapper):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = model_name
        
        # RL variables
        self.rl_group= config.args.get("rl_group")
        self.num_of_clusters = self.config.args.get("num_of_clusters", 100)
        self.map_sampling_dist = config.args.get("sampling_dist", 20)
        self.rl_group_size = 50 # Number of RLs (estimation) should be replaced by the actual number of RLs
        self.model_structure = [3, 32, 64, None, self.rl_group_size]
    
        # Sequence variables(revisit)
        self.timestep = 1 #(15 min)
        self.time_lag_h = config.lag
        self.input_time_len_h = self.config.args.get("input_time_len_h", 12) # 12 hours default
        self.seq_h = self.input_time_len_h * 4 #4 timesteps(15min) per hour
            

    def create_dataset(self):
        self.train_data_manager = CNNDataManager(self.config.batch_size, self.input_time_len_h, self.time_lag_h, self.rl_group, self.timestep, self.map_sampling_dist, self.num_of_clusters)
        self.rl_group_size = self.train_data_manager.rl_group_size
        model_structure = self.model_structure
        model_structure[-1] = self.rl_group_size
        self.model_structure = model_structure
        self.test_data_manager = CNNDataManager(None, self.input_time_len_h, self.time_lag_h, self.rl_group, self.timestep, self.map_sampling_dist, self.num_of_clusters, test_mode=True)
        
    def init_model(self):
        logger.info(f"Initializing CNN1D model")
        self.create_dataset()
        device = check_device()
        self.model = ConvoModel2RLs(self.model_structure, self.seq_h).to(device)
        self.model.float()
        self.loss_function = nn.MSELoss()
        self.eval_loss_function = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        logger.info(f"Model initialized")
        return True 
        
    def train(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        train_loader, val_loader  = self.train_data_manager.idx_prep(batch_size=self.config.batch_size, shuffle=True, random_seed=321)
        device = check_device()
        model = self.model

        history = {
            "loss": [],
            "eval_loss": [],
            "val_loss": [],
            "eval_val_loss": [],
            "train_time": None
        }
        
        train_start = time.time()
        for epoch in range(self.config.epochs):
            losses = []
            eval_losses = []
            val_losses = []
            eval_val_losses = []
            for input_in_batch, output_in_batch  in train_loader:
                input_in_batch = input_in_batch.to(device)
                output_in_batch = output_in_batch.to(device)
                self.model.train()
                self.optimizer.zero_grad()
                pred = model(input_in_batch.float(), None, output_in_batch.float(), False)
                output_in_batch = output_in_batch.float()
                loss = self.loss_function(pred, output_in_batch)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                eval_losses.append(self.eval_loss_function(pred, output_in_batch).detach().cpu().numpy())
                logger.info(f"Batch train loss: {losses[-1]} eval loss: {eval_losses[-1]}")
        
            with torch.no_grad():
                for input_in_batch, output_in_batch in val_loader:
                    input_in_batch = input_in_batch.to(device)
                    output_in_batch = output_in_batch.to(device)
                    self.model.eval()
                    pred_val = model(input_in_batch.float(), None, output_in_batch.float(), False)
                    output_in_batch = output_in_batch.float()
                    val_loss = self.loss_function(pred_val, output_in_batch)
                    val_losses.append(val_loss.item())
                    eval_val_losses.append(self.eval_loss_function(pred_val, output_in_batch).detach().cpu().numpy())
                    logger.info(f"Batch validation loss: {val_losses[-1]} eval loss: {eval_val_losses[-1]}")
            
            mean_loss = np.mean(losses)
            mean_val_loss = np.mean(val_losses)
            mean_eval_loss = np.mean(eval_losses)
            mean_eval_val_loss = np.mean(eval_val_losses)
            history["loss"].append(mean_loss)
            history["eval_loss"].append(mean_val_loss)
            history["val_loss"].append(mean_eval_loss)
            history["eval_val_loss"].append(mean_eval_val_loss)
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Train Loss: {mean_loss}, Validation Loss: {mean_val_loss}, Eval Train Loss: {mean_eval_loss}, Eval Validation Loss: {mean_eval_val_loss}")
            
            #Add early stopping logic here
            if epoch > 5 and mean_val_loss < 0.01:
                logger.info("Early stopping condition met")
                self.model = model
                break
            
        self.model = model
        train_end = time.time()
        train_time = train_end - train_start
        history["train_time"] = train_time
        logger.info(f"Training completed in {train_end - train_start} seconds")
        model_file = self.save_model_checkpoint(self.config.run_id, run_dir, self.model, self.config)
        logger.info(f"Model saved to {model_file}")
        return history, train_time, model_file
        
    def predict(self):
        test_inputs = self.test_data_manager.test_inputs
        test_ref = self.test_data_manager.rl_dataset_prep() 
        
        with torch.no_grad():
            self.model.eval()
            device = check_device()
            test_inputs = test_inputs.to(device)
            start_time = time.time()
            y_hat = self.model(test_inputs.float(), None, None, False)
            end_time = time.time()
            prediction_time = end_time - start_time
            logger.info(f"Prediction completed in {prediction_time} seconds")
            mse = self.loss_function(y_hat, test_ref.to(device).float()).item()
            rmse = np.sqrt(mse)
            logger.info(f"Final loss: {mse}, RMSE: {rmse}")
            return mse, rmse, prediction_time
            
    def validate_model(self):
        mse, rmse, prediction_time = self.predict()
        return mse, rmse, prediction_time
    
    def save_model_checkpoint(self, run_id, run_dir, model, config):
        model_file = os.path.join(run_dir, f"{config.model_name}_{run_id}_model.pt")
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'sampling_dist': self.map_sampling_dist,
                'num_of_clusters': self.num_of_clusters,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'model_strcture': self.model_structure,
                'num_epochs': self.config.epochs,
                'run_id': run_id,
            }, os.path.join(run_dir, model_file))
            return model_file
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
        return None
        
class ConvoModel2RLs(nn.Module):
    
    def __init__(self, model_structure, seq_h):
        #change kernal size and pooling size
        super(ConvoModel2RLs, self).__init__()
        convo_1_kernel = 4
        convo_2_kernel = 2  # Reduced from 4 to 2
        pool_1_kernel = 3
        
        # Add padding to preserve spatial dimensions
        self.convo_1 = nn.Conv1d(model_structure[0], model_structure[1], convo_1_kernel, padding=1)
        self.pooling_1 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        self.convo_2 = nn.Conv1d(model_structure[1], model_structure[1], convo_2_kernel, padding=1)
        self.pooling_2 = nn.MaxPool1d(pool_1_kernel, ceil_mode=True)
        
        # Update dimension calculation to account for padding
        self.dim_past_convo = lambda dim_in, kernel, padding: int(np.ceil((dim_in - kernel + 2*padding + 1)/pool_1_kernel))
        flat_dim = self.dim_past_convo(
            self.dim_past_convo(seq_h, convo_1_kernel, 1), 
            convo_2_kernel, 1
        ) * model_structure[1]
        self.hidden_1 = nn.Linear(flat_dim, model_structure[-3])
        self.lyr_out = nn.Linear(model_structure[-3], model_structure[-1])
        self.lrelu = nn.LeakyReLU()

    def forward(self, src, dw_filter, ref, trainning_with_dw_mask=True):
        src = self.convo_1(src.transpose(1, 2))
        src = torch.tanh(self.pooling_1(src))
        src = self.convo_2(src)
        src = self.lrelu(self.pooling_2(src))
        src = self.lrelu(self.hidden_1(src.view(src.size()[0], 1, -1)))
        src = self.lyr_out(src).squeeze(1)
        if trainning_with_dw_mask:
            src = ref * dw_filter + (-dw_filter + 1) * src + torch.relu(src*dw_filter - ref * dw_filter)
        return src