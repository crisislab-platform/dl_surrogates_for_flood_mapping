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
    output_dir = os.path.join(RUN_DIR, model, "hyperparam_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tuning_metrics_summary_file = os.path.join(output_dir, "tuning_metrics_summary.csv")
    if os.path.exists(tuning_metrics_summary_file):
        tuning_metrics_summary = pd.read_csv(tuning_metrics_summary_file)
    
    else:
        tuning_metrics_summary = pd.DataFrame(columns=['hyperparameters', 'average_val_loss'])
        
    tuning_metrics = pd.read_csv(os.path.join(RUN_DIR, model, "tuning_metrics.csv"))
    hyperparams = tuning_metrics['hyperparameters'].unique()
    # Now write to a new CSV file the average val loss with the hyperparamater combination
    for idx, hyperparam in enumerate(hyperparams):
        hyperparam_metrics = tuning_metrics[tuning_metrics['hyperparameters'] == hyperparam]
        average_loss, average_epocs = get_average_cvloss(model, hyperparam_metrics, idx)
        new_row = {
            'hyper_param_id': idx,
            'hyperparameters': hyperparam,
            'average_val_loss': average_loss,
            'average_convergence_epoch': round(average_epocs),
        }
        if tuning_metrics_summary.size == 0:
            tuning_metrics_summary = pd.DataFrame([new_row])
        else:
            tuning_metrics_summary = pd.concat([tuning_metrics_summary, pd.DataFrame([new_row])], ignore_index=True)
    tuning_metrics_summary.to_csv(tuning_metrics_summary_file, index=False)
    logger.info(f"Hyperparameter tuning metrics saved to {tuning_metrics_summary_file}")
    
def get_average_cvloss(model, metrics, hyperparam_id):
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
    

    output_dir = os.path.join(RUN_DIR, model, "hyperparam_analysis")
    avg_epochs = sum(epochs)/len(epochs)
    folds = 8
    
    logger.info(f"Hyperparams id {hyperparam_id}")
    logger.info(f"Average best epoch across {len(epochs)} folds: {avg_epochs:.2f}")
    average_val_loss = sum(val_losses) / len(val_losses)
    logger.info(f"Hyperparams {hyperparam_id} - Average val loss across {len(val_losses)} folds: {average_val_loss}")
    logger.info(f"Loss plots saved to {output_dir}")
    return average_val_loss, avg_epochs

def create_plot1(folds, epochs, avg_epochs, model, hyperparam_id, output_dir):
    # Create a summary plot of best epochs
    plt.figure(figsize=(10, 6))
    plt.plot(folds, epochs, 'o-', color='green', linewidth=2, markersize=8, label='Best Epoch')

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
    
def create_plot2(folds, val_losses, avg_val_loss, model, hyperparam_id, output_dir):
    # Create a summary plot of all validation losses
    plt.figure(figsize=(10, 6))

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
    
def get_memory_usage(train_memory_usage, inference_memory_usage):
    # Parse memory usage information
    inference_memory_usage = inference_memory_usage.split(",")
    train_memory_usage = train_memory_usage.split(",")
    
    train_gpu = float(train_memory_usage[0].split(":")[1].strip())
    train_cpu = float(train_memory_usage[1].split(":")[1].strip())
    
    pred_gpu_ = float(inference_memory_usage[0].split(":")[1].strip())
    pred_cpu = float(inference_memory_usage[1].split(":")[1].strip())
    
    # Convert to GB
    return train_gpu, train_cpu, pred_gpu_, pred_cpu
        
            
    
def metrics()-> pd.DataFrame:
    logger.info(f"Quality and Efficiency metrics for all models")
    perf_metrics_file  = os.path.join(RUN_DIR, "final_performance_metrics.csv")
    training_metrics_file = os.path.join(RUN_DIR, "final_training_metrics.csv")
    
    if not os.path.exists(perf_metrics_file) or not os.path.exists(training_metrics_file):
        logger.error(f"Metrics files do not exist")
    train_metrics_df = pd.read_csv(training_metrics_file)
    perf_metrics_df = pd.read_csv(perf_metrics_file)
   
    model_metrics = pd.DataFrame(columns=['model_name', 'rmse', 'nse', 'mrmse', 'flops', 'params', 'inference_latency', 'inference_memory_usage', 'training_memory_usage', 'train_time', 'total_neurons'])
    # For each model in the metrics file read metrics
    for idx, train_metrics in train_metrics_df.iterrows():
            perf_metrics = perf_metrics_df[(perf_metrics_df['run_id'] == train_metrics['run_id'])]
            if perf_metrics.empty:
                logger.warning(f"No performance metrics found for model {train_metrics['model']}")
                continue
            perf_metrics = perf_metrics.iloc[0]
            model  = train_metrics['model']
            
            # I get the memory usages in the following way
            # 'max_cuda_memory': 357310398464.0, 'max_cpu_m...  410.930379 
            # Need to read the max_cuda_memory and max_cpu_memory from the perf_metrics
            # Parse memory usage information
            train_memory_usage = train_metrics['training_memory_usage']
            inference_memory_usage = perf_metrics['pred_memory_usage']
            
            train_gpu, train_cpu, pred_gpu, pred_cpu = get_memory_usage(train_memory_usage, inference_memory_usage)
            
            row = {
                'model_name': model,
                'rmse': perf_metrics['mse'], 
                'nse': perf_metrics['nse'],
                'mrmse': perf_metrics['mRMSE'],
                'flops': perf_metrics['flops'],
                'inference_latency': perf_metrics['inference_latency'],
                'inference_gpu': pred_gpu,
                'inference_cpu': pred_cpu,
                'params': train_metrics['trainable_params'],
                'training_gpu': train_gpu,
                'training_cpu': train_cpu,
                'training_memory_usage': train_metrics['training_memory_usage'],
                'train_time': train_metrics['train_time'],
                'total_neurons': train_metrics['total_neurons'],
                'hyperparameters': train_metrics['hyperparameters'],
            }
            # Append the row to the DataFrame
            model_metrics = pd.concat([model_metrics, pd.DataFrame([row])], ignore_index=True)
            
    return model_metrics

