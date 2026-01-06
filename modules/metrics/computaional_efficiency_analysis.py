import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modules.lib.constants import RUN_DIR, OUTPUT_DIR
from modules.lib.constants import USRR_CNN1D_COMBINED, PICNN1D_V1, SRR_LSTM_COMBINED, CNN1D_V1, HDL_FM_V1
from modules.metrics.topsis_analysis import topsis

from modules.lib.constants import RMSE, MRMSE, HITRATE, CSI, F2SCORE, F3SCORE, INFERENCE_TIMES, INFERENCE_MEMORY_USAGE, FLOPS, PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FOOTPRINT_METRICS = os.path.join(OUTPUT_DIR, "footprint_metrics")

model_colors = {
    "Tier-1": "#0173B2", 
    "Tier-2": "#DE8F05",   
    "Tier-3": "#029E73",       
    "Tier-4": "#D55E00",      
}

def get_quality_and_footprint_metrics():
    # Generate and save comparative performance visualizations of different models
    training_metrics_file = os.path.join(RUN_DIR, "final_training_metrics.csv")
    performance_metrics_file = os.path.join(RUN_DIR, "final_performance_metrics.csv")
    
    # Copy these files to OUTPUT_DIR for record keeping
    os.makedirs(FOOTPRINT_METRICS, exist_ok=True)
    os.system(f"cp {training_metrics_file} {OUTPUT_DIR}")
    os.system(f"cp {performance_metrics_file} {OUTPUT_DIR}")
    
    try:
        train_metrics = pd.read_csv(training_metrics_file)
        # Avoid model SLTM-SRR as it is not a valid model
        train_metrics = train_metrics[train_metrics['model'] != SRR_LSTM_COMBINED]
        training_metrics = train_metrics.groupby('model').last().reset_index()
        perf_metrics = pd.read_csv(performance_metrics_file)
        
        # Filter performance metrics to only include models in training metrics
        perf_metrics = perf_metrics[perf_metrics['run_id'].isin(training_metrics['run_id'])]
        training_metrics = training_metrics[training_metrics['run_id'].isin(perf_metrics['run_id'])]
        
        perf_metrics = perf_metrics.sort_values(by='model')
        training_metrics = training_metrics.sort_values(by='model')
        
        model_names = perf_metrics['model'].values
        rmse = perf_metrics['rmse'].values
        mrmse = perf_metrics['mRMSE'].values
        csi = perf_metrics['csi'].values
        hit_rate = perf_metrics['hit_rate'].values
        f2_score = perf_metrics['f2_score'].values
        f3_score = perf_metrics['f3_score'].values
        inference_times = perf_metrics['inference_latency'].values
        params = training_metrics['trainable_params'].values
        flops = perf_metrics['flops'].values
   

        gflops = [round(float(flop / 1e9), 1) for flop in flops]  # Convert to GFLOPs with 1 decimal place
        params = [round(param / 1e6, 2) for param in params]  # Convert to Millions with 2 decimal places
        
        # Extract inference memory usage
        inference_memory_usage = []
        for inference_memory in perf_metrics['pred_memory_usage'].values:
            if isinstance(inference_memory, str) and inference_memory.startswith('{'):
                try:
                    mem_dict = eval(inference_memory)
                    inference_memory_usage.append(mem_dict.get('max_cuda_memory', 0) / (1024**3))
                except:
                    inference_memory_usage.append(0)
            else:
                inference_memory_usage.append(0)
                
             
        metrics_dict = {
            RMSE: rmse, 
            MRMSE: mrmse, 
            CSI: csi, 
            HITRATE: hit_rate,
            F2SCORE: f2_score, 
            F3SCORE: f3_score,
            INFERENCE_TIMES: inference_times,
            PARAMS: params,
            FLOPS: gflops,
            INFERENCE_MEMORY_USAGE: inference_memory_usage
        }
        
        # Save final metrics to OUTPUT_DIR
        final_metrics_file = os.path.join(OUTPUT_DIR, "final_metrics_summary.csv")
        ouput_df = pd.DataFrame(metrics_dict)
        ouput_df['model'] = model_names
        ouput_df.to_csv(final_metrics_file, index=False)
        # Add model names to the final metrics file
        logger.info(f"Saved final metrics summary to {final_metrics_file}")
        
        return model_names, metrics_dict
    
    except Exception as e:
        logger.error(f"Error reading metrics files: {e}")
        return None, None
    
def computational_efficiency():
    os.makedirs(FOOTPRINT_METRICS, exist_ok=True)
    
    model_names, metrics_dict = get_quality_and_footprint_metrics()
    
    logger.info(f"Generating performance comparisons for {len(model_names)} models")
    
    create_performance_plot(metrics_dict[FLOPS], metrics_dict[RMSE], metrics_dict[INFERENCE_TIMES], model_names,
                            'GFLOPs', 'RMSE (m)', 'Inference Time (s)',
                            '(a) FLOPs vs RMSE vs Inference Latency', 'flops_vs_rmse_vs__inference_time.png')
    
    create_performance_plot(metrics_dict[PARAMS], metrics_dict[RMSE], metrics_dict[INFERENCE_MEMORY_USAGE], model_names,
                    'Parameters (Million)', 'RMSE (m)', 'Maximum GPU Memory Usage (GB)',
                    '(b) Model Parameters vs RMSE vs Max GPU Memory', 'parameters_vs_rmse_vs_memory.png')

    topsis(model_names, metrics_dict)
    logger.info("Completed generating performance visualizations and topsis analysis.")
    
    
def create_performance_plot(x_values, y_values, z_values, model_names, x_label, y_label, z_label, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set scientific color scheme - white background with dark elements
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Rename model_names to standard values
    for i, model in enumerate(model_names):
        if model == USRR_CNN1D_COMBINED:
            model_names[i] = "Tier-1"
        elif model == PICNN1D_V1:
            model_names[i] = "Tier-3"
        elif model == SRR_LSTM_COMBINED:
            model_names[i] = "SRR-LSTM"
        elif model == CNN1D_V1:
            model_names[i] = "Tier-2"
        elif model == HDL_FM_V1:
            model_names[i] = "Tier-4"
    
    # Convert input values to numeric format if they aren't already
    numeric_x_values = []
    numeric_y_values = []
    numeric_z_values = []
    
    for x_val, y_val, z_val in zip(x_values, y_values, z_values):
        # Convert x value if needed
        if isinstance(x_val, str):
            try:
                if x_val.startswith('{'):
                    mem_dict = eval(x_val)
                    numeric_x = mem_dict.get('max_cuda_memory', 0) / (1024**3)
                else:
                    numeric_x = float(x_val)
            except:
                logger.warning(f"Could not convert x value: {x_val} to numeric, using 0")
                numeric_x = 0
        else:
            numeric_x = float(x_val) if x_val is not None else 0
            
        # Convert y value if needed
        if isinstance(y_val, str):
            try:
                if y_val.startswith('{'):
                    mem_dict = eval(y_val)
                    numeric_y = mem_dict.get('max_cuda_memory', 0) / (1024**3)
                else:
                    numeric_y = float(y_val)
            except:
                logger.warning(f"Could not convert y value: {y_val} to numeric, using 0")
                numeric_y = 0
        else:
            numeric_y = float(y_val) if y_val is not None else 0
        
        # Convert z value if needed
        if isinstance(z_val, str):
            try:
                if z_val.startswith('{'):
                    mem_dict = eval(z_val)
                    numeric_z = mem_dict.get('max_cuda_memory', 0) / (1024**3)
                else:
                    numeric_z = float(z_val)
            except:
                logger.warning(f"Could not convert z value: {z_val} to numeric, using 100")
                numeric_z = 100
        else:
            numeric_z = float(z_val) if z_val is not None else 100
        
        numeric_x_values.append(numeric_x)
        numeric_y_values.append(numeric_y)
        numeric_z_values.append(numeric_z)
    
    # Normalize z_values to appropriate bubble sizes
    def normalize_bubble_sizes(values, min_size=100, max_size=800):
        if max(values) == min(values):
            return [min_size] * len(values)
        normalized = [(val - min(values)) / (max(values) - min(values)) for val in values]
        return [min_size + (max_size - min_size) * norm for norm in normalized]
    
    bubble_sizes = normalize_bubble_sizes(numeric_z_values)
    

    # Assign colors based on model names, fallback to tab10 colors if model not in mapping
    colors = []
    for model in model_names:
        if model in model_colors:
            colors.append(model_colors[model])
        elif model in ["USRR-1DCNN", "PI1DCNN", "SRR-LSTM", "1DCNN", "HDL-FM"]:
            colors.append(model_colors[model])
        else:
            colors.append('#0066CC')  # Default blue for unknown models
    
    # Plot each point with bubble sizes based on z_values
    for i, model in enumerate(model_names):
        ax.scatter(
            numeric_x_values[i], 
            numeric_y_values[i],
            s=bubble_sizes[i],
            marker='o',
            color=colors[i],
            alpha=0.8,
            # edgecolors='black',
            # linewidth=2,
            label=model
        )
        
        # Mark the center of the bubble with a small black dot
        ax.scatter(
            numeric_x_values[i], 
            numeric_y_values[i],
            s=20,
            marker='o',
            color='black',
            alpha=1.0,
            zorder=10
        )
    
    
    # Add model name labels with arrows - colored for better readability
    for i, model in enumerate(model_names):
        
        if model == "Tier-4":
            x_offset = -60
            y_offset = 30
            
        else:
            x_offset = 50
            y_offset = -20
            
        ax.annotate(model, 
                  xy=(numeric_x_values[i], numeric_y_values[i]),
                  xytext=(x_offset, y_offset),
                  textcoords='offset points',
                  fontsize=12,
                  fontweight='bold',
                  color='white',  # White text for better contrast
                  bbox=dict(boxstyle="round,pad=0.3", fc=colors[i], ec="black", alpha=0.9),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=colors[i], lw=1.5))
        
        if model == 'Tier-4':
            y_offset = 16
            x_offset = -10
        else: 
            x_offset = 0
            y_offset = -16
        
        #Annotate 3rd dimension as well
        ax.annotate(f'{numeric_z_values[i]:.2f}', 
                  xy=(numeric_x_values[i], numeric_y_values[i]),
                  xytext=(x_offset, y_offset),
                  textcoords='offset points',
                  fontsize=8,
                  fontweight='bold',
                  color='white',  # White text for better contrast
                  bbox=dict(boxstyle="round,pad=0.3", fc=colors[i], ec="black", alpha=0.9))
    
    # Add simple bubble size legend - italic text below the diagram on bottom left
    ax.text(0.5, -0.15, f'Bubble Size = {z_label}', 
            transform=ax.transAxes, fontsize=12, fontweight='normal', fontstyle='italic', 
            color='black', ha='center')
    
    # Configure plot with scientific styling
    ax.set_xlabel(x_label, fontsize=12, color='black')
    ax.set_ylabel(y_label, fontsize=12, color='black')
    
    ax.grid(True, linestyle='-', alpha=0.3, color='gray', linewidth=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(colors='black', which='both')
    
    ax.set_facecolor("#e0eae0")  # Light gray background
    fig.patch.set_facecolor("#e0eae0")  # White figure background
    
    # Save the figure
    output_file = os.path.join(FOOTPRINT_METRICS, filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.title(title, fontsize=14, pad=10)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_file}")
    plt.close()
  