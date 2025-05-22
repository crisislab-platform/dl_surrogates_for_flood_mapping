from modules.metrics_reader.metrics_reader import metrics
import logging
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects  # Add proper import for path effects
import numpy as np
import os
from modules.lib.constants import GRAPH_OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetricsViz")


def create_flops_inference_time_plot(metric_df):
    metric_df['flops'] = metric_df['flops'].astype(float)/1000000 # in M
    metric_df['inference_latency'] = metric_df['inference_latency'].astype(float) # in s
    metric_df['rmse'] = metric_df['rmse'].astype(float)  # Ensure RMSE is float
    
    # FLOPS in the x-axis and inference time in the y-axis and model name for the points
    plt.figure(figsize=(12, 9))
    
    # Create scatter plot with RMSE as color
    scatter = plt.scatter(
        metric_df['flops'], 
        metric_df['inference_latency'], 
        c=metric_df['rmse'],  # Color by RMSE
        cmap='viridis_r',  # Reversed colormap so lower RMSE (better) is darker
        s=120,  # Slightly larger points for better visibility
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add colorbar to show RMSE scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('RMSE (lower is better)', rotation=270, labelpad=20)
    
    # Add model names as labels for each point
    for i, model in enumerate(metric_df['model_name']):
        plt.annotate(
            model, 
            (metric_df['flops'].iloc[i], metric_df['inference_latency'].iloc[i]),
            xytext=(7, 7), 
            textcoords='offset points',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
    
    
    plt.title('Model Performance: FLOPS vs Inference Time vs RMSE')
    plt.xlabel('FLOPS (millions)')
    plt.ylabel('Inference Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    
    # Ensure output directory exists
    os.makedirs(os.path.join(GRAPH_OUTPUT_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, 'plots', 'flops_inference_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created enhanced performance plot at {os.path.join(GRAPH_OUTPUT_DIR, 'plots', 'flops_inference_rmse.png')}")

def create_flops_rmse_plot(metric_df):
    # Set the style for a more professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    
    metric_df['flops'] = metric_df['flops'].astype(float)
    metric_df['inference_latency'] = metric_df['inference_latency'].astype(float)  # in s
    metric_df['rmse'] = metric_df['rmse'].astype(float)  # Ensure RMSE is float
    
    # Calculate size based on FLOPS - scale appropriately
    min_size = 1000  # Increased from 1
    max_size = 2000  # Increased from 10
    min_flops = metric_df['flops'].min()
    max_flops = metric_df['flops'].max()
    
    logger.info(f"Min FLOPS: {min_flops}, Max FLOPS: {max_flops}")
    
    # If only one point or all FLOPS are the same, use a default size
    if min_flops == max_flops:
        sizes = [500] * len(metric_df)  # Increased default size
    else:
        # Scale FLOPS to size range
        # Use logarithmic scaling if the range is large
        flops_range = max_flops - min_flops
        if max_flops / (min_flops + 0.001) > 100:
            # Log scaling for wide ranges
            sizes = min_size + (np.log(metric_df['flops'] + 0.001) - np.log(min_flops + 0.001)) * (max_size - min_size) / (np.log(max_flops + 0.001) - np.log(min_flops + 0.001))
        else:
            # Linear scaling for narrower ranges
            sizes = min_size + (metric_df['flops'] - min_flops) * (max_size - min_size) / flops_range
    
    # Add debug info for the calculated sizes
    for i, (model, size, flop) in enumerate(zip(metric_df['model_name'], sizes, metric_df['flops'])):
        logger.info(f"Model: {model}, FLOPS: {flop:.2f}M, Bubble Size: {size:.2f}")
    
    # Create the plot with improved aesthetics
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f1f3f5')
    
    # Use a professional blue palette instead of viridis
    blue_colors = ['#0466c8', '#0353a4', '#023e7d', '#002855', '#001845', '#001233']
    # Ensure we have enough colors by cycling if needed
    blues = [blue_colors[i % len(blue_colors)] for i in range(len(metric_df))]
    
    # Create scatter plot with blue color scheme
    scatter = ax.scatter(
        metric_df['inference_latency'], 
        metric_df['rmse'], 
        s=sizes,
        c=blues,  # Use blue colors
        alpha=0.8,
        edgecolors='white',
        linewidth=1.5,
        zorder=10
    )

    # Add model names as labels with clean styling
    for i, model in enumerate(metric_df['model_name']):
        ax.annotate(
            model,
            (metric_df['inference_latency'].iloc[i], metric_df['rmse'].iloc[i]),
            xytext=(10, 30), 
            textcoords='offset points',
            fontsize=15,
            fontweight='bold',
            color='white',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='#333333')],
            bbox=dict(
                boxstyle="round,pad=0.4,rounding_size=0.2",
                fc=blues[i],
                ec='white',
                alpha=0.9,
                lw=1,
                zorder=15
            ),
            ha='center', va='center'
        )
    
    # Create a cleaner, more professional legend at the top right
    # Create a small legend with just min and max values
    legend_handles = []
    
    # Add two circles for min and max FLOPS
    min_circle = plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#0466c8', markersize=8, 
                      label=f'Min: {min_flops:.1f}M')
    max_circle = plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#0466c8', markersize=16, 
                      label=f'Max: {max_flops:.1f}M')
    
    legend_handles.extend([min_circle, max_circle])
    
    # Create the legend
    legend = ax.legend(handles=legend_handles,
                    loc='upper right',
                    title='FLOPS',
                    frameon=True,
                    fontsize=13,
                    title_fontsize=15,
                    facecolor='white',
                    edgecolor='#dee2e6',
                    framealpha=0.95,
                    borderpad=1,
                    handletextpad=1.5)
    legend.get_frame().set_linewidth(1)
    
    
    # Add stylish grid
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, color='#ced4da', zorder=0)
    
    # Improve title and labels
    ax.set_title('Model Performance Analysis', 
               fontsize=18, fontweight='bold', pad=20,
               color='#212529')
    
    ax.set_xlabel('Inference Time (seconds)', 
                fontsize=15, fontweight='bold', labelpad=15,
                color='#343a40')
    
    ax.set_ylabel('RMSE (m)', 
                fontsize=15, fontweight='bold', labelpad=15,
                color='#343a40')
    
    # Add a rectangle as a border
    for spine in ax.spines.values():
        spine.set_color('#ced4da')
        spine.set_linewidth(1.5)
    
    # Increase font size for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=13)  # Increase tick font size
    
    # Add a subtitle
    plt.figtext(0.5, 0.01, 
              "Bubble size represents computational complexity (FLOPS)",
              ha='center', fontsize=14, fontstyle='italic', color='#495057')
    
    # Ensure tight layout and margins
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Ensure output directory exists
    os.makedirs(os.path.join(GRAPH_OUTPUT_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, 'plots', 'model_performance_analysis.png'), 
               dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    logger.info(f"Created enhanced model performance analysis plot at {os.path.join(GRAPH_OUTPUT_DIR, 'plots', 'model_performance_analysis.png')}")


def plot_metrics():
    metric_df  = metrics()
    # 1. Create the FLOPS and inference time.  FLOPS in the x-axis and inference time in the y-axis
    create_flops_inference_time_plot(metric_df)
    create_flops_rmse_plot(metric_df)
