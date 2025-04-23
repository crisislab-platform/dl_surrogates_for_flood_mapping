from modules.lib.constants import RUN_DIR
from modules.lib.constants import USRR_UNET_V1, USRR_1DCNN_V1, USRR_CNN1D_COMBINED, CNN1D_V1
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_metrics(row):
    logger.info(f"Model: {row['model']}")
    logger.info(f"Flops: {row['flops']}")
    # logger.info(f"Params: {row['trainable_params']}")
    # logger.info(f"Train Loss: {row['loss']}")
    # logger.info(f"Validation Loss: {row['val_loss']}")
    logger.info(f"RMSE: {row['pred_rmse']}")
    # logger.info(f"RMSE Wet: {row['pred_rmse_wet']}")
    # logger.info(f"Train Time: {row['train_time']}")
    # logger.info(f"Pred Time: {row['pred_time']}")
    logger.info(f"Wet Accuracy:{row['pred_acc_wet']}")
    logger.info(f"Row :{row}")
    logger.info(f"Neurons: {row['total_neurons']}")
    logger.info(f"Trainable Params: {row['trainable_params']}")
    logger.info(f"Memory Usage: {row['training_memory_usage']}")
    logger.info(f"==" * 50) 
    

def read_metrics():
    metrics_file = f'{RUN_DIR}/training_metrics.csv'
    ussr_1dcnn_metafile = os.path.join(RUN_DIR, USRR_1DCNN_V1, "run_metadata.csv")
    
    metrics_df = pd.read_csv(metrics_file)
    cnn_1d_stdalone = metrics_df[metrics_df['model'] == CNN1D_V1].sort_index(ascending=False).iloc[0]
    
    # For each sampling_dist find the latest run
  
    log_metrics(cnn_1d_stdalone)
    
    unet_metrics = pd.read_csv(os.path.join(RUN_DIR, USRR_UNET_V1, "run_metadata.csv"))
    unet_model = unet_metrics.groupby('sampling_dist')
    unet_models = {}
    for sampling_dist, group in unet_model:
        latest_run = group.sort_values('run_id', ascending=False).iloc[0]['run_id']
        unet_models[sampling_dist] = metrics_df[metrics_df['run_id'] == latest_run].iloc[0]
        
    logger.info("Flops and params for each model")
    
    for sampling_dist, unet_model in unet_models.items():
        log_metrics(unet_model)
    
    # Read the USRR_1DCNN_V1 metadata file
    usrr_1dcnn_metadata = pd.read_csv(ussr_1dcnn_metafile)
    sampling_dist = usrr_1dcnn_metadata['sampling_dist'].unique()
    clusters = usrr_1dcnn_metadata['cluster_size'].unique()
    
    # For each sampling_dist find the latest run
    usrr_1dcnn_models = {}
    for sampling_dist_value in sampling_dist:  # Ensure correct iteration variable
        usrr_1dcnn_models[sampling_dist_value] = {}
        for cluster in clusters:
            group = usrr_1dcnn_metadata[
                (usrr_1dcnn_metadata['sampling_dist'] == sampling_dist_value) & 
                (usrr_1dcnn_metadata['cluster_size'] == cluster)
            ]
            if group.empty:  # Check if the group is empty
                logger.warning(f"No entries found for sampling_dist {sampling_dist_value} and cluster {cluster}")
                continue
            for i in range(cluster):
                cluster_entries = group[group['rl_group'] == i]
                if cluster_entries.empty:  # Check if cluster_entries is empty
                    logger.warning(f"No entries found for rl_group {i} in sampling_dist {sampling_dist_value} and cluster {cluster}")
                    continue
                latest_run = cluster_entries.sort_values('run_id', ascending=False).iloc[0]['run_id']
                row = metrics_df[metrics_df['run_id'] == latest_run]
                if row.empty:  # Check if the row is empty
                    logger.warning(f"No metrics found for run_id {latest_run}")
                    continue
                if cluster not in usrr_1dcnn_models[sampling_dist_value]:
                    usrr_1dcnn_models[sampling_dist_value][cluster] = []
                usrr_1dcnn_models[sampling_dist_value][cluster].append(row.iloc[0])
                
    usrr_cnn_metrics = {}
    logger.info("Flops and params for USRR_1DCNN_V1")
    for sampling_dist_value, clusters_dict in usrr_1dcnn_models.items():
        usrr_cnn_metrics[sampling_dist_value] = {}
        for cluster, models in clusters_dict.items():
            total_flops = 0
            total_nurons = 0
            parameters = 0
            for i, model in enumerate(models):
                total_flops += model['flops']
                total_nurons += model['total_neurons']
                parameters += model['trainable_params']
                # log_metrics(model)
            usrr_cnn_metrics[sampling_dist_value][cluster] = {'flops': total_flops, 'total_nurons': total_nurons, 'parameters': parameters}
       
    combined_metrics = {}         
    for sampling_dist_value, clusters_dict in usrr_cnn_metrics.items():
        for cluster, cluster_metrics in clusters_dict.items():
            if sampling_dist_value not in combined_metrics:
                combined_metrics[sampling_dist_value] = {}
            unet_metric = unet_models[sampling_dist_value]
            combined_metrics[sampling_dist_value][cluster] = {'flops': unet_metric['flops'] + cluster_metrics['flops'], 
                                                                'total_nurons': unet_metric['total_neurons'] + cluster_metrics['total_nurons'], 'parameters': unet_metric['trainable_params'] + cluster_metrics['parameters']}
            
            logger.info(f"Total Flops for sampling dist {sampling_dist_value}, cluster {cluster}: {combined_metrics[sampling_dist_value][cluster]['flops']}")
            logger.info(f"Total Neurons for sampling dist {sampling_dist_value}, cluster {cluster}: {combined_metrics[sampling_dist_value][cluster]['total_nurons']}")
            logger.info(f"Total Parameters for sampling dist {sampling_dist_value}, cluster {cluster}: {combined_metrics[sampling_dist_value][cluster]['parameters']}")
    
    # Read the USRR_CNN1D_COMBINED metadata file
    usrr_combined_model = metrics_df[metrics_df['model'] == USRR_CNN1D_COMBINED].sort_index(ascending=False).iloc[0]
    log_metrics(usrr_combined_model)

