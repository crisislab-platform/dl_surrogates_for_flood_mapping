from modules.lib.constants import RUN_DIR
from modules.lib.constants import USRR_UNET_V1, USRR_1DCNN_V1, USRR_CNN1D_COMBINED, CNN1D_V1
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

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
    

# def read_metrics():
#     metrics_file = f'{RUN_DIR}/training_metrics.csv'
#     ussr_1dcnn_metafile = os.path.join(RUN_DIR, USRR_1DCNN_V1, "run_metadata.csv")
    
#     metrics_df = pd.read_csv(metrics_file)
#     cnn_1d_stdalone = metrics_df[metrics_df['model'] == CNN1D_V1].sort_index(ascending=False).iloc[0]
    
#     # For each sampling_dist find the latest run
  
#     log_metrics(cnn_1d_stdalone)
    
#     unet_metrics = pd.read_csv(os.path.join(RUN_DIR, USRR_UNET_V1, "run_metadata.csv"))
#     unet_model = unet_metrics.groupby('sampling_dist')
#     unet_models = {}
#     for sampling_dist, group in unet_model:
#         latest_run = group.sort_values('run_id', ascending=False).iloc[0]['run_id']
#         unet_models[sampling_dist] = metrics_df[metrics_df['run_id'] == latest_run].iloc[0]
        
#     logger.info("Flops and params for each model")
    
#     for sampling_dist, unet_model in unet_models.items():
#         log_metrics(unet_model)
    
#     # Read the USRR_1DCNN_V1 metadata file
#     usrr_1dcnn_metadata = pd.read_csv(ussr_1dcnn_metafile)
#     sampling_dist = usrr_1dcnn_metadata['sampling_dist'].unique()
#     clusters = usrr_1dcnn_metadata['cluster_size'].unique()
    
#     # For each sampling_dist find the latest run
#     usrr_1dcnn_models = {}
#     for sampling_dist_value in sampling_dist:  # Ensure correct iteration variable
#         usrr_1dcnn_models[sampling_dist_value] = {}
#         for cluster in clusters:
#             group = usrr_1dcnn_metadata[
#                 (usrr_1dcnn_metadata['sampling_dist'] == sampling_dist_value) & 
#                 (usrr_1dcnn_metadata['cluster_size'] == cluster)
#             ]
#             if group.empty:  # Check if the group is empty
#                 logger.warning(f"No entries found for sampling_dist {sampling_dist_value} and cluster {cluster}")
#                 continue
#             for i in range(cluster):
#                 cluster_entries = group[group['rl_group'] == i]
#                 if cluster_entries.empty:  # Check if cluster_entries is empty
#                     logger.warning(f"No entries found for rl_group {i} in sampling_dist {sampling_dist_value} and cluster {cluster}")
#                     continue
#                 latest_run = cluster_entries.sort_values('run_id', ascending=False).iloc[0]['run_id']
#                 row = metrics_df[metrics_df['run_id'] == latest_run]
#                 if row.empty:  # Check if the row is empty
#                     logger.warning(f"No metrics found for run_id {latest_run}")
#                     continue
#                 if cluster not in usrr_1dcnn_models[sampling_dist_value]:
#                     usrr_1dcnn_models[sampling_dist_value][cluster] = []
#                 usrr_1dcnn_models[sampling_dist_value][cluster].append(row.iloc[0])
                
#     usrr_cnn_metrics = {}
#     logger.info("Flops and params for USRR_1DCNN_V1")
#     for sampling_dist_value, clusters_dict in usrr_1dcnn_models.items():
#         usrr_cnn_metrics[sampling_dist_value] = {}
#         for cluster, models in clusters_dict.items():
#             total_flops = 0
#             total_nurons = 0
#             parameters = 0
#             for i, model in enumerate(models):
#                 total_flops += model['flops']
#                 total_nurons += model['total_neurons']
#                 parameters += model['trainable_params']
#                 # log_metrics(model)
#             usrr_cnn_metrics[sampling_dist_value][cluster] = {'flops': total_flops, 'total_nurons': total_nurons, 'parameters': parameters}
       
#     combined_metrics = {}         
#     for sampling_dist_value, clusters_dict in usrr_cnn_metrics.items():
#         for cluster, cluster_metrics in clusters_dict.items():
#             if sampling_dist_value not in combined_metrics:
#                 combined_metrics[sampling_dist_value] = {}
#             unet_metric = unet_models[sampling_dist_value]
#             combined_metrics[sampling_dist_value][cluster] = {'flops': unet_metric['flops'] + cluster_metrics['flops'], 
#                                                                 'total_nurons': unet_metric['total_neurons'] + cluster_metrics['total_nurons'], 'parameters': unet_metric['trainable_params'] + cluster_metrics['parameters']}
            
#             logger.info(f"Total Flops for sampling dist {sampling_dist_value}, cluster {cluster}: {combined_metrics[sampling_dist_value][cluster]['flops']}")
#             logger.info(f"Total Neurons for sampling dist {sampling_dist_value}, cluster {cluster}: {combined_metrics[sampling_dist_value][cluster]['total_nurons']}")
#             logger.info(f"Total Parameters for sampling dist {sampling_dist_value}, cluster {cluster}: {combined_metrics[sampling_dist_value][cluster]['parameters']}")
    
#     # Read the USRR_CNN1D_COMBINED metadata file
#     usrr_combined_model = metrics_df[metrics_df['model'] == USRR_CNN1D_COMBINED].sort_index(ascending=False).iloc[0]
#     log_metrics(usrr_combined_model)


def hyperparam_analysis(model):
    logger.info(f"Hyper parameter analysis for {model}")
    metrics = pd.read_csv(os.path.join(RUN_DIR, model, "tuning_metrics.csv"))
    hyperparams = metrics['hyperparameters'].unique()
    
    tuning_metrics_summary_file = os.path.join(RUN_DIR, model, "tuning_metrics_summary.csv")
    if os.path.exists(tuning_metrics_summary_file):
        tuning_metrics_summary = pd.read_csv(tuning_metrics_summary_file)
    
    else:
        tuning_metrics_summary = pd.DataFrame(columns=['hyperparameters', 'average_val_loss'])
    # Now write to a new CSV file the average val loss with the hyperparamater combination
    for idx, hyperparam in enumerate(hyperparams):
        hyperparam_metrics = metrics[metrics['hyperparameters'] == hyperparam]
        average_loss = get_average_loss(model, hyperparam_metrics, idx)
        new_row = {
            'hyper_param_id': idx,
            'hyperparameters': hyperparam,
            'average_val_loss': average_loss,
        }
        if tuning_metrics_summary.size == 0:
            tuning_metrics_summary = pd.DataFrame([new_row])
        else:
            tuning_metrics_summary = pd.concat([tuning_metrics_summary, pd.DataFrame([new_row])], ignore_index=True)
    tuning_metrics_summary.to_csv(tuning_metrics_summary_file, index=False)
    
def get_average_loss(model, metrics, hyperparam_id):
    val_losses = []
    metrics.sort_values(by=['fold'], inplace=True)
    metrics.reset_index(drop=True, inplace=True)
    epochs = []
    
    for index, row in metrics.iterrows(): 
        fold = row['fold']
        loss_str = row['loss']
        val_loss_str = row['val_loss']
        best_epoch = row['best_epoch']
        best_val_loss = row['best_val_rmse']
        val_losses.append(best_val_loss)
        epochs.append(best_epoch + 1)
    
    # Create a summary plot of all validation losses
    output_dir = os.path.join(RUN_DIR, model, "hyperparam_analysis")
    plt.figure(figsize=(10, 6))
    folds = range(1, len(val_losses) + 1)
    plt.plot(folds, val_losses, 'o-', color='blue', linewidth=2, markersize=8, label='Validation Loss')
    plt.axhline(y=sum(val_losses)/len(val_losses), color='r', linestyle='--', 
               label=f'Average: {sum(val_losses)/len(val_losses):.4f}')
    plt.title(f'Best Validation Loss by Fold - {model}\nHyperparams: {hyperparam_id}')
    plt.xlabel('Fold')
    plt.ylabel('Best Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)  # Ensure all fold numbers are shown on x-axis
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'validation_loss_summary_{hyperparam_id}.png'))
    plt.close()
    
    # Create a summary plot of best epochs
    plt.figure(figsize=(10, 6))
    plt.plot(folds, epochs, 'o-', color='green', linewidth=2, markersize=8, label='Best Epoch')
    avg_epochs = sum(epochs)/len(epochs)
    plt.axhline(y=avg_epochs, color='r', linestyle='--', 
               label=f'Average: {avg_epochs:.2f}')
    plt.title(f'Best Epoch by Fold - {model}\nHyperparams: {hyperparam_id}')
    plt.xlabel('Fold')
    plt.ylabel('Best Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'best_epoch_summary_{hyperparam_id}.png'))
    plt.close()
    
    logger.info(f"Average best epoch across {len(epochs)} folds: {avg_epochs:.2f}")
    
    average_val_loss = sum(val_losses) / len(val_losses)
    logger.info(f"Hyperparams {hyperparam_id} - Average val loss across {len(val_losses)} folds: {average_val_loss}")
    logger.info(f"Loss plots saved to {output_dir}")
    return average_val_loss