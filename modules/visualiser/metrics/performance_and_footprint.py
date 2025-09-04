import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Make sure this import is at the module level
from modules.lib.constants import RUN_DIR, GRAPH_OUTPUT_DIR
from modules.lib.constants import USRR_CNN1D_COMBINED, PICNN1D_V1, SRR_LSTM_COMBINED, CNN1D_V1, HDL_FM_V1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(GRAPH_OUTPUT_DIR, "perf_plots")


def create_performance_plot(x_values, y_values, z_values, model_names, x_label, y_label, z_label, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set scientific color scheme - white background with dark elements
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Rename model_names to standard values
    for i, model in enumerate(model_names):
        if model == USRR_CNN1D_COMBINED:
            model_names[i] = "USRR-1DCNN"
        elif model == PICNN1D_V1:
            model_names[i] = "PI1DCNN"
        elif model == SRR_LSTM_COMBINED:
            model_names[i] = "SRR-LSTM"
        elif model == CNN1D_V1:
            model_names[i] = "1DCNN"
        elif model == HDL_FM_V1:
            model_names[i] = "HDL-FM"
    
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
    
    # Use blue for all bubbles
    colors = ['#0066CC'] * len(model_names)  # Blue for all bubbles
    
    # Plot each point with bubble sizes based on z_values
    for i, model in enumerate(model_names):
        ax.scatter(
            numeric_x_values[i], 
            numeric_y_values[i],
            s=bubble_sizes[i],
            marker='o',
            color=colors[i],
            alpha=0.8,
            edgecolors='black',
            linewidth=2,
            label=model
        )
    
    # Add model name labels with arrows - black text
    for i, model in enumerate(model_names):
        ax.annotate(model, 
                  xy=(numeric_x_values[i], numeric_y_values[i]),
                  xytext=(20, 20),
                  textcoords='offset points',
                  fontsize=12,
                  fontweight='bold',
                  color='black',
                  bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="black", alpha=0.9),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black', lw=1.5))
    
    # Add simple bubble size legend - italic text below the diagram on bottom left
    ax.text(0.5, -0.18, f'Bubble Size = {z_label}', 
            transform=ax.transAxes, fontsize=10, fontweight='normal', fontstyle='italic', 
            color='black', ha='center')
    
    # Configure plot with scientific styling
    ax.set_xlabel(x_label, fontsize=12, color='black', fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, color='black', fontweight='bold')
    
    ax.grid(True, linestyle='-', alpha=0.3, color='gray', linewidth=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(colors='black', which='both')
    
    # Save the figure
    output_file = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # plt.title(title, fontsize=12, pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_file}")
    plt.close()
    
def draw_memory_usage_plot(train_memory_usage, pred_memory_usage, model_names, rmse):
    """Generate and save a memory usage comparison plot with RMSE on y-axis."""
    
    # memory usage is in the following format
    # "{'max_cuda_memory': 1856871424.0, 'max_cpu_memory': 618255808.0, 'total_cuda_memory': 1856008704.0, 'total_cpu_memory': 4.0, 'total_cpu_time': 1546069.375, 'total_gpu_time': 1365034.25}"#
    
    # Extract max_cuda_memory and max_cpu_memory from the memory usage data
    train_cuda_memory = []
    train_cpu_memory = []
    pred_cuda_memory = []
    pred_cpu_memory = []
    
    for mem in train_memory_usage:
        if isinstance(mem, str) and mem.startswith('{'):
            try:
                # Convert string representation of dict to actual dict
                mem_dict = eval(mem)
                # Convert bytes to GB
                train_cuda_memory.append(mem_dict.get('max_cuda_memory', 0) / (1024**3))
                train_cpu_memory.append(mem_dict.get('max_cpu_memory', 0) / (1024**3))
            except:
                train_cuda_memory.append(mem)  # Fallback to original value
                train_cpu_memory.append(0)
        else:
            train_cuda_memory.append(mem)  # Assume it's already the CUDA memory in GB
            train_cpu_memory.append(0)
    
    for mem in pred_memory_usage:
        if isinstance(mem, str) and mem.startswith('{'):
            try:
                mem_dict = eval(mem)
                pred_cuda_memory.append(mem_dict.get('max_cuda_memory', 0) / (1024**3))
                pred_cpu_memory.append(mem_dict.get('max_cpu_memory', 0) / (1024**3))
            except:
                pred_cuda_memory.append(mem)  # Fallback to original value
                pred_cpu_memory.append(0)
        else:
            pred_cuda_memory.append(mem)  # Assume it's already the CUDA memory in GB
            pred_cpu_memory.append(0)
    
    # Use CUDA memory for the plot
    train_memory = train_cuda_memory
    pred_memory = pred_cuda_memory
    
    # Create enhanced scatter plot with quadrants
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use consistent marker shape but different colors for models
    colors = plt.cm.tab10(range(len(model_names)))
    marker_size = 200  # Fixed size for all markers
    
    # Plot each point with the same marker shape but different colors
    for i, model in enumerate(model_names):
        ax.scatter(
            pred_memory[i], 
            rmse[i],
            s=marker_size,
            marker='o',  # Use circles for all models
            color=colors[i],
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5,
            label=model
        )
    
    # Add quadrant lines and labels (based on median values to create 4 regions)
    med_x = np.median(pred_memory)
    med_y = np.median(rmse)
    
    # Get current axis limits to ensure complete coverage
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Ensure the limits include all data points with some padding
    x_padding = 0.05 * (max(pred_memory) - min(pred_memory))
    y_padding = 0.05 * (max(rmse) - min(rmse))
    
    x_min = min(pred_memory) - x_padding
    y_min = min(rmse) - y_padding
    x_max = max(pred_memory) + x_padding
    y_max = max(rmse) + y_padding
    
    # Set these expanded limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add quadrant dividers
    ax.axvline(med_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(med_y, color='gray', linestyle='--', alpha=0.5)
    
    # Add colored quadrants with properly filled regions extending to the axes
    # Bottom left (best) - low memory, low error
    ax.fill_between(
        [x_min, med_x],      # x-range
        [y_min, y_min],      # bottom
        [med_y, med_y],      # top
        color="palegreen", alpha=0.2
    )
    
    # Bottom right - high memory, low error
    ax.fill_between(
        [med_x, x_max],      # x-range
        [y_min, y_min],      # bottom
        [med_y, med_y],      # top
        color="khaki", alpha=0.2
    )
    
    # Top left - low memory, high error
    ax.fill_between(
        [x_min, med_x],      # x-range
        [med_y, med_y],      # bottom
        [y_max, y_max],      # top
        color="khaki", alpha=0.2
    )
    
    # Top right (worst) - high memory, high error
    ax.fill_between(
        [med_x, x_max],      # x-range
        [med_y, med_y],      # bottom
        [y_max, y_max],      # top
        color="lightcoral", alpha=0.2
    )
    
    # Create legend handles for quadrants with updated colors
    from matplotlib.patches import Patch
    quadrant_handles = [
        Patch(facecolor="palegreen", edgecolor="green", alpha=0.7, label="BEST: Low Memory, Low Error"),
        Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, label="High Memory, Low Error"),
        Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, label="Low Memory, High Error"),
        Patch(facecolor="lightcoral", edgecolor="red", alpha=0.7, label="WORST: High Memory, High Error")
    ]
    
    # Add model name labels with arrows for clearer association
    for i, model in enumerate(model_names):
        ax.annotate(model, 
                  xy=(pred_memory[i], rmse[i]),
                  xytext=(20, 20),  # Offset text by fixed amount
                  textcoords='offset points',
                  fontsize=12,
                  fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", fc=colors[i], ec="black", alpha=0.2),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='gray'))
    
    # Get handles and labels for the models
    handles, labels = ax.get_legend_handles_labels()
    
    # Create two-row legend: models in first row, quadrants in second row
    first_legend = ax.legend(handles, labels, 
                           loc='upper center', bbox_to_anchor=(0.5, -0.10), 
                           ncol=len(model_names) if len(model_names) <= 4 else 4,
                           frameon=True, fancybox=True, shadow=True,
                           title="Models")
    
    # Add second legend for quadrants
    ax.add_artist(first_legend)
    second_legend = ax.legend(handles=quadrant_handles, 
                            loc='upper center', bbox_to_anchor=(0.5, -0.25),
                            ncol=2, 
                            frameon=True, fancybox=True, shadow=True,
                            title="Quadrants")
    
    # Configure plot with consistent formatting
    ax.set_xlabel('Inference Memory Usage (GB)', fontsize=12)
    ax.set_ylabel('RMSE (m)', fontsize=12)
    
    # Add grid and improve aesthetics
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure
    output_file = os.path.join(OUTPUT_DIR, 'model_memory_performance_quadrant.png')
    plt.title('Model Memory Usage vs Performance', fontsize=12, pad=20)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved enhanced scatter plot visualization to {output_file}")
    plt.close()
    
def draw_radar_chart(model_names, rmse, inference_times, pred_memory, flops, params):
    
    for i, model in enumerate(model_names):
        if model == USRR_CNN1D_COMBINED:
            model_names[i] = "USRR-1DCNN"
        elif model == PICNN1D_V1:
            model_names[i] = "PI1DCNN"
        elif model == SRR_LSTM_COMBINED:
            model_names[i] = "SRR-LSTM"
        elif model == CNN1D_V1:
            model_names[i] = "1DCNN"
        elif model == HDL_FM_V1:
            model_names[i] = "HDL-FM"
    
    """Create a comprehensive radar chart comparing all metrics."""
    try:
        # Use consistent colors for models
        colors = plt.cm.tab10(range(len(model_names)))
        
        # Convert all inputs to numeric values
        def ensure_numeric(values):
            result = []
            for val in values:
                if isinstance(val, str):
                    try:
                        if val.startswith('{'):
                            # Handle memory usage dictionaries
                            mem_dict = eval(val)
                            result.append(mem_dict.get('max_cuda_memory', 0) / (1024**3))
                        else:
                            result.append(float(val))
                    except:
                        logger.warning(f"Could not convert value: {val} to numeric, using 1.0")
                        result.append(1.0)  # Use 1.0 instead of 0 to avoid division by zero
                else:
                    result.append(float(val) if val is not None else 1.0)
            return result
        
        # Convert all metrics to numeric values
        numeric_rmse = ensure_numeric(rmse)
        numeric_inference_times = ensure_numeric(inference_times)
        numeric_pred_memory = ensure_numeric(pred_memory)
        numeric_flops = ensure_numeric(flops)
        numeric_params = ensure_numeric(params)
        
        
        weights = {
            'E_RMSE': 1,  # Model accuracy (inverse of RMSE)
            'E_Latency': 1,  # Inference speed (inverse of inference time)
            'E_Memory': 0.1,  # Memory efficiency (inverse of memory usage)
            'E_FLOPs': 0.01,  # Computation efficiency (inverse of FLOPs)
            'E_Parameters': 0.01  # Model size efficiency (inverse of parameter count)
        }

        def normalize_and_invert(values, metric):
            """
            Logarithmic normalization and inversion for radar chart visualization.
            
            Mathematical Formula:
            For a set of values X = {x₁, x₂, ..., xₙ} where all xᵢ > 0:
            
            Step 1: Logarithmic transformation
            Y = {ln(x₁), ln(x₂), ..., ln(xₙ)}
            
            Step 2: Min-max normalization in log space
            Yₙₒᵣₘ = (ln(xᵢ) - min(Y)) / (max(Y) - min(Y))
            
            Step 3: Inversion for "higher is better" visualization
            E_metric = 1 - Yₙₒᵣₘ
            
            Combined equation:
            E_metric = 1 - (ln(xᵢ) - ln(xₘᵢₙ)) / (ln(xₘₐₓ) - ln(xₘᵢₙ))
            
            Simplified form using logarithm properties:
            E_metric = 1 - ln(xᵢ/xₘᵢₙ) / ln(xₘₐₓ/xₘᵢₙ)
            
            Where:
            - xᵢ is the original metric value for model i
            - xₘᵢₙ = min(X), xₘₐₓ = max(X) across all models
            - E_metric ∈ [0,1] with higher values indicating better performance
            - Fallback to linear normalization if any xᵢ ≤ 0
            """
            max_val = max(values)
            min_val = min(values)
            if max_val == min_val:
                return [1.0] * len(values)
            if max_val == 0 or min_val <= 0:
                # Fallback to linear normalization if log not possible
                values = [(v - min_val) / (max_val - min_val) for v in values]
                return [1 - v for v in values]
            
            # Use log scale normalization for better sensitivity to small differences
            log_values = [np.log(v) for v in values]
            log_min = min(log_values)
            log_max = max(log_values)
            
            if log_max == log_min:
                return [1.0] * len(values)
                
            # Normalize log values to [0, 1]
            norm_log_values = [(v - log_min) / (log_max - log_min) for v in log_values]
            
            # Invert for radar chart (higher is better)
            return [1 - v for v in norm_log_values]
        
        #normalise before inverting
        norm_rmse = normalize_and_invert(numeric_rmse, 'E_RMSE')  # Normalize RMSE
        norm_inf_time = normalize_and_invert(numeric_inference_times, 'E_Latency')  # Normalize inference time
        norm_pred_memory = normalize_and_invert(numeric_pred_memory, 'E_Memory')  # Normalize memory
        norm_flops = normalize_and_invert(numeric_flops, 'E_FLOPs')  # Normalize FLOPs
        norm_params = normalize_and_invert(numeric_params, 'E_Parameters')  # Normalize parameters
        
        # Apply weights to the normalized and inverted values
        # norm_rmse = [v * weights['E_RMSE'] for v in norm_rmse]
        # norm_inf_time = [v * weights['E_Latency'] for v in norm_inf_time]
        # norm_pred_memory = [v * weights['E_Memory'] for v in norm_pred_memory]
        # norm_flops = [v * weights['E_FLOPs'] for v in norm_flops]
        # norm_params = [v * weights["E_Parameters"] for v in norm_params]
        
        # #Now invert the normalized values for radar chart (avoid any values becoming 0)
        # norm_rmse = [(1 / (v + 0.1)) * weights['E_RMSE'] for v in norm_rmse]  # Invert RMSE
        # norm_inf_time = [(1 / (v + 0.1)) * weights['E_Latency'] for v in norm_inf_time]  # Invert inference time
        # norm_pred_memory = [(1 / (v + 0.1)) * weights['E_Memory'] for v in norm_pred_memory]  # Invert memory
        # norm_flops = [(1 / (v + 0.1)) * weights['E_FLOPs'] for v in norm_flops]  # Invert FLOPs
        # norm_params = [(1 / (v + 0.1)) * weights["E_Parameters"] for v in norm_params]  # Invert parameters
        
        
        
        # Calculate the area of each model's radar chart
        def calculate_area(values):
            # Convert the values to cartesian coordinates using angles on a unit circle
            angles = np.linspace(0, 2 * np.pi, len(values) + 1)[:-1]  # Exclude the last angle (2π)
            
            # Calculate x,y coordinates for each point on the radar chart
            x = [values[i] * np.cos(angles[i]) for i in range(len(values))]
            y = [values[i] * np.sin(angles[i]) for i in range(len(values))]
            
            # Add the first point at the end to close the polygon
            x.append(x[0])
            y.append(y[0])
            
            # Calculate area using the shoelace formula
            area = 0.5 * np.abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(values))))
            return area
        
        model_values = [[norm_rmse[i], norm_inf_time[i], norm_pred_memory[i], norm_flops[i], norm_params[i]]
            for i in range(len(model_names)) ]
        model_area = {}
        for i, model in enumerate(model_names):
            area = calculate_area(model_values[i])
            model_area[model] = area
        
                
        #save normalized values and the area of each model to a csv file
        df = pd.DataFrame({
            'Model': model_names,
            'E_RMSE': norm_rmse,
            'E_Latency': norm_inf_time,
            'E_Memory': norm_pred_memory,
            'E_FLOPs': norm_flops,
            'E_Parameters': norm_params,
            'Area': [model_area[model] for model in model_names]
        })
        #save to csv
        output_csv = os.path.join(OUTPUT_DIR, 'radar_chart_data.csv')
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        # Create radar chart data with all metrics - use more descriptive labels
        categories = [
            'E$_{RMSE}$',
            'E$_{Latency}$',
            'E$_{FLOPs}$',
            'E$_{Parameters}$',
            'E$_{Memory}$'
        ]
        
        # Create descriptions for the metrics legend
        metric_descriptions = [
            ('A: Model Accuracy', 'Higher is better - Inverse of RMSE error'),
            ('B: Inference Speed', 'Higher is better - Inverse of inference time'),
            ('C: Memory Efficiency', 'Higher is better - Inverse of memory usage'),
            ('D: Computation Efficiency', 'Higher is better - Inverse of FLOPs'),
            ('E: Model Size Efficiency', 'Higher is better - Inverse of parameter count')
        ]
        
        # Create single figure with a larger size to accommodate side legends
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        # Use more vibrant custom colors instead of default tab10
        custom_colors = [
            '#1E88E5',  # Blue
            '#D81B60',  # Magenta
            '#FFC107',  # Amber
            '#004D40',  # Teal
            '#8E24AA',  # Purple
            '#FB8C00',  # Orange
            '#5D4037',  # Brown
            '#2E7D32',  # Green
            '#F44336',  # Red
            '#3949AB',  # Indigo
        ]
        # Ensure we have enough colors by cycling if needed
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(model_names))]
        
        # Number of categories
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add background grid with more visible concentric circles
        ax.set_facecolor('#f8f8f8')
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            circle = plt.Circle((0, 0), level, fill=False, color='gray', 
                             linewidth=0.5, alpha=0.5)
            ax.add_patch(circle)
            
        # Draw level labels on one of the axes
        ax.text(0, 0.2, '0.2', transform=ax.transData, ha='center', va='bottom', fontsize=8, color='gray')
        ax.text(0, 0.4, '0.4', transform=ax.transData, ha='center', va='bottom', fontsize=8, color='gray')
        ax.text(0, 0.6, '0.6', transform=ax.transData, ha='center', va='bottom', fontsize=8, color='gray')
        ax.text(0, 0.8, '0.8', transform=ax.transData, ha='center', va='bottom', fontsize=8, color='gray')
        ax.text(0, 1.0, '1.0', transform=ax.transData, ha='center', va='bottom', fontsize=8, color='gray')
        
        # Draw the radar chart for each model
        for i, model in enumerate(model_names):
            values = [
                norm_rmse[i], 
                norm_inf_time[i], 
                norm_pred_memory[i], 
                norm_flops[i], 
                norm_params[i]
            ]
            values += values[:1]  # Close the loop
            
            # Plot values with thicker lines
            ax.plot(angles, values, linewidth=2.5, linestyle='solid', 
                   label=model, color=colors[i], zorder=10)
            ax.fill(angles, values, alpha=0.25, color=colors[i], zorder=5)
            
            # Add markers at each data point
            ax.scatter(angles[:-1], values[:-1], s=80, 
                      color=colors[i], edgecolor='white', linewidth=1, zorder=15)
        
        # Set category labels with enhanced styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        
        # Add padding to move tick labels slightly away from the circle
        ax.tick_params(axis='x', pad=25)
        
        # Remove radial labels and set grid
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
        
        # Set title
        # ax.set_title('Model Performance Comparison', fontsize=14, pad=20)
        
        # Move the polar plot to the left side to make space for legends
        plt.subplots_adjust(right=0.7)  # Adjusted for better spacing
        
        # Create a text box for models on the right side - closer to the plot
        handles, labels = ax.get_legend_handles_labels()
        model_legend = fig.legend(handles, labels, 
                  loc='upper right', 
                  frameon=True, fancybox=True, shadow=True,
                  title="Models",
                  title_fontsize=14,
                  fontsize=12)
        
        # Add the model legend to the figure
        fig.add_artist(model_legend)
    
        logger.info(f"Created radar chart with {len(model_names)} models")
        logger.info(f"Model areas: {model_area}")
        # Save with a much larger bbox to ensure annotations are included
        output_file = os.path.join(OUTPUT_DIR, 'comprehensive_model_comparison_radar.png')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        logger.info(f"Saved enhanced radar chart visualization to {output_file}")
        plt.close()
        
        #Add a model vs efficieny plot
        draw_model_efficiency_plot(model_names, model_area, rmse, inference_times, pred_memory, flops, params)
        
    except Exception as e:
        logger.error(f"Error creating comprehensive radar chart: {e}")
        import traceback
        logger.error(traceback.format_exc())

def draw_model_efficiency_plot(model_names, model_areas, rmse=None, inference_times=None, pred_memory=None, flops=None, params=None):
    """Create an accuracy vs computational demand plot."""
    try:
        # Rename model names to standard values
        for i, model in enumerate(model_names):
            if model == USRR_CNN1D_COMBINED:
                model_names[i] = "USRR-1DCNN"
            elif model == PICNN1D_V1:
                model_names[i] = "PI1DCNN"
            elif model == SRR_LSTM_COMBINED:
                model_names[i] = "SRR-LSTM"
            elif model == CNN1D_V1:
                model_names[i] = "1DCNN"
            elif model == HDL_FM_V1:
                model_names[i] = "HDL-FM"
        
        # Check if all required data is provided
        if not all(param is not None for param in [rmse, inference_times, pred_memory, flops, params]):
            logger.warning("Not all required data provided for efficiency plot. Skipping.")
            return
        
        # Convert all inputs to numeric values
        def ensure_numeric(values):
            result = []
            for val in values:
                if isinstance(val, str):
                    try:
                        if val.startswith('{'):
                            mem_dict = eval(val)
                            result.append(mem_dict.get('max_cuda_memory', 0) / (1024**3))
                        else:
                            result.append(float(val))
                    except:
                        logger.warning(f"Could not convert value: {val} to numeric, using 1.0")
                        result.append(1.0)
                else:
                    result.append(float(val) if val is not None else 1.0)
            return result
        
        numeric_rmse = ensure_numeric(rmse)
        numeric_inference_times = ensure_numeric(inference_times)
        numeric_pred_memory = ensure_numeric(pred_memory)
        numeric_flops = ensure_numeric(flops)
        numeric_params = ensure_numeric(params)
        
        # Calculate accuracy (inverse of RMSE, normalized)
        max_rmse = max(numeric_rmse)
        min_rmse = min(numeric_rmse)
        if max_rmse == min_rmse:
            accuracy = [1.0] * len(numeric_rmse)
        else:
            accuracy = [1 - (rmse_val - min_rmse) / (max_rmse - min_rmse) for rmse_val in numeric_rmse]
        
        # Calculate computational demand
        def normalize_metric(values):
            max_val = max(values)
            min_val = min(values)
            if max_val == min_val:
                return [0.5] * len(values)
            return [(val - min_val) / (max_val - min_val) for val in values]
        
        norm_inference_times = normalize_metric(numeric_inference_times)
        norm_pred_memory = normalize_metric(numeric_pred_memory)
        norm_flops = normalize_metric(numeric_flops)
        norm_params = normalize_metric(numeric_params)
        
        computational_demand = []
        for i in range(len(model_names)):
            demand = (norm_inference_times[i] + norm_pred_memory[i] + norm_flops[i] + norm_params[i]) / 4
            computational_demand.append(demand)
        
        # Create single figure for scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot scatter plot
        colors_scatter = plt.cm.tab10(range(len(model_names)))
        marker_size = 300
        
        for i, model in enumerate(model_names):
            ax.scatter(computational_demand[i], accuracy[i], s=marker_size, marker='o',
                      color=colors_scatter[i], alpha=0.8, edgecolors='black', linewidth=2, label=model)
        
        # # Add quadrant analysis
        # med_x = np.median(computational_demand)
        # med_y = np.median(accuracy)
        
        # x_padding = 0.05 * (max(computational_demand) - min(computational_demand))
        # y_padding = 0.05 * (max(accuracy) - min(accuracy))
        
        # x_min = min(computational_demand) - x_padding
        # x_max = max(computational_demand) + x_padding
        # y_min = min(accuracy) - y_padding
        # y_max = max(accuracy) + y_padding
        
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        
        # # Add quadrant dividers and colors
        # ax.axvline(med_x, color='gray', linestyle='--', alpha=0.5)
        # ax.axhline(med_y, color='gray', linestyle='--', alpha=0.5)
        
        # ax.fill_between([x_min, med_x], [y_min, y_min], [med_y, med_y], color="khaki", alpha=0.3)
        # ax.fill_between([med_x, x_max], [y_min, y_min], [med_y, med_y], color="lightcoral", alpha=0.3)
        # ax.fill_between([x_min, med_x], [med_y, med_y], [y_max, y_max], color="palegreen", alpha=0.3)
        # ax.fill_between([med_x, x_max], [med_y, med_y], [y_max, y_max], color="khaki", alpha=0.3)
        
        # Create quadrant legend
        # from matplotlib.patches import Patch
        # quadrant_handles = [
        #     Patch(facecolor="palegreen", edgecolor="green", alpha=0.7, 
        #           label="BEST: High Accuracy, Low Computational Demand"),
        #     Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, 
        #           label="High Accuracy, High Computational Demand"),
        #     Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, 
        #           label="Low Accuracy, Low Computational Demand"),
        #     Patch(facecolor="lightcoral", edgecolor="red", alpha=0.7, 
        #           label="WORST: Low Accuracy, High Computational Demand")
        # ]
        
        # Add model annotations
        for i, model in enumerate(model_names):
            ax.annotate(model, xy=(computational_demand[i], accuracy[i]), xytext=(20, 20),
                       textcoords='offset points', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc=colors_scatter[i], ec="black", alpha=0.2),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='gray'))
        
        # Configure scatter plot
        ax.set_xlabel('Computational Footprint', fontsize=12, fontweight='bold')
        ax.set_ylabel('Quality', fontsize=12, fontweight='bold')
        # ax.set_title('Model Accuracy vs Computational Demand\n(Efficiency-Performance Trade-off)', 
                    # fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Create legends
        handles, labels = ax.get_legend_handles_labels()
        first_legend = ax.legend(handles, labels, 
                               loc='upper center', bbox_to_anchor=(0.5, -0.10), 
                               ncol=len(model_names) if len(model_names) <= 4 else 4,
                               frameon=True, fancybox=True, shadow=True,
                               title="Models")
        
        ax.add_artist(first_legend)
        # second_legend = ax.legend(handles=quadrant_handles, 
        #                         loc='upper center', bbox_to_anchor=(0.5, -0.25),
        #                         ncol=2, 
        #                         frameon=True, fancybox=True, shadow=True,
        #                         title="Performance Regions")
        
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(OUTPUT_DIR, 'accuracy_vs_computational_demand.png')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved accuracy vs computational demand plot to {output_file}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating model efficiency plot: {e}")
        import traceback
        logger.error(traceback.format_exc())

def create_multi_dimensional_comparison(model_names, rmse, inference_times, flops, params, inference_memory_usage):
    """Create a landscape subplot with three 3D scatter plots."""
    try:
        # Rename model names to standard values
        display_names = []
        for model in model_names:
            if model == USRR_CNN1D_COMBINED:
                display_names.append("USRR-1DCNN")
            elif model == PICNN1D_V1:
                display_names.append("PI1DCNN")
            elif model == SRR_LSTM_COMBINED:
                display_names.append("SRR-LSTM")
            elif model == CNN1D_V1:
                display_names.append("1DCNN")
            elif model == HDL_FM_V1:
                display_names.append("HDL-FM")
            else:
                display_names.append(model)
        
        # Convert all inputs to numeric values
        def ensure_numeric(values):
            result = []
            for val in values:
                if isinstance(val, str):
                    try:
                        if val.startswith('{'):
                            mem_dict = eval(val)
                            result.append(mem_dict.get('max_cuda_memory', 0) / (1024**3))
                        else:
                            result.append(float(val))
                    except:
                        logger.warning(f"Could not convert value: {val} to numeric, using 1.0")
                        result.append(1.0)
                else:
                    result.append(float(val) if val is not None else 1.0)
            return result
        
        numeric_rmse = ensure_numeric(rmse)
        numeric_inference_times = ensure_numeric(inference_times)
        numeric_flops = ensure_numeric(flops)
        numeric_params = ensure_numeric(params)
        numeric_memory_usage = ensure_numeric(inference_memory_usage)
        
        # Calculate compound computational footprint (normalized sum of all metrics including memory)
        def normalize_metric(values):
            max_val = max(values)
            min_val = min(values)
            if max_val == min_val:
                return [0.5] * len(values)
            return [(val - min_val) / (max_val - min_val) for val in values]
        
        norm_inference_times = normalize_metric(numeric_inference_times)
        norm_flops = normalize_metric(numeric_flops)
        norm_params = normalize_metric(numeric_params)
        norm_memory_usage = normalize_metric(numeric_memory_usage)
        
        compound_footprint = []
        for i in range(len(display_names)):
            footprint = (norm_inference_times[i] + norm_flops[i] + norm_params[i] + norm_memory_usage[i]) / 4
            compound_footprint.append(footprint)
        
        # Create figure with 1x3 subplots (landscape)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Use consistent colors for all plots
        colors = plt.cm.tab10(range(len(display_names)))
        
        # Helper function to normalize bubble sizes
        def normalize_bubble_sizes(values, min_size=100, max_size=800):
            if max(values) == min(values):
                return [min_size] * len(values)
            normalized = [(val - min(values)) / (max(values) - min(values)) for val in values]
            return [min_size + (max_size - min_size) * norm for norm in normalized]
        
        # Plot 1: RMSE vs Inference Time with FLOPS as bubble size
        ax1 = axes[0]
        bubble_sizes_1 = normalize_bubble_sizes(numeric_flops)
        
        for i, model in enumerate(display_names):
            ax1.scatter(numeric_inference_times[i], numeric_rmse[i], 
                       s=bubble_sizes_1[i], color=colors[i], alpha=0.7, 
                       edgecolors='black', linewidth=1.5, label=model)
        
        ax1.set_xlabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
        ax1.set_title('RMSE vs Inference Time\n(Bubble size = FLOPs)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for subplot 1
        for i, model in enumerate(display_names):
            ax1.annotate(model, 
                        xy=(numeric_inference_times[i], numeric_rmse[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc=colors[i], ec="black", alpha=0.3),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.1", color='gray'))
        
        # Plot 2: Parameters vs RMSE with FLOPS as bubble size
        ax2 = axes[1]
        bubble_sizes_2 = normalize_bubble_sizes(numeric_flops)
        
        for i, model in enumerate(display_names):
            ax2.scatter(numeric_params[i], numeric_rmse[i], 
                       s=bubble_sizes_2[i], color=colors[i], alpha=0.7, 
                       edgecolors='black', linewidth=1.5, label=model)
        
        ax2.set_xlabel('Parameters (Million)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
        ax2.set_title('RMSE vs Parameters\n(Bubble size = FLOPs)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for subplot 2
        for i, model in enumerate(display_names):
            ax2.annotate(model, 
                        xy=(numeric_params[i], numeric_rmse[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc=colors[i], ec="black", alpha=0.3),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.1", color='gray'))
        
        # Plot 3: Compound Computational Footprint vs RMSE with same bubble size
        ax3 = axes[2]
        bubble_sizes_3 = [300] * len(display_names)  # Same size for all
        
        for i, model in enumerate(display_names):
            ax3.scatter(compound_footprint[i], numeric_rmse[i], 
                       s=bubble_sizes_3[i], color=colors[i], alpha=0.7, 
                       edgecolors='black', linewidth=1.5, label=model)
        
        ax3.set_xlabel('Compound Computational Footprint', fontsize=12, fontweight='bold')
        ax3.set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
        ax3.set_title('RMSE vs Computational Footprint\n(Equal bubble sizes)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add annotations for subplot 3
        for i, model in enumerate(display_names):
            ax3.annotate(model, 
                        xy=(compound_footprint[i], numeric_rmse[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc=colors[i], ec="black", alpha=0.3),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.1", color='gray'))
        
        # Remove individual legends and create a single legend for all subplots
        for ax in axes:
            ax.legend().set_visible(False)
        
        # Create a single legend below all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, 
                  loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=len(display_names), frameon=True, fancybox=True, shadow=True,
                  title="Models")
        
        # Add bubble size legends for the first two plots
        # Bubble legend for plots 1 and 2 (FLOPs)
        flops_min, flops_max = min(numeric_flops), max(numeric_flops)
        bubble_legend_sizes = [100, 300, 600]
        bubble_legend_flops = [flops_min, (flops_min + flops_max) / 2, flops_max]
        
        # Add text explanation for bubble sizes
        fig.text(0.02, 0.95, f'Bubble Size Legend (Plots 1-2):\nSmall = {flops_min:.1f}B FLOPs\nMedium = {(flops_min + flops_max) / 2:.1f}B FLOPs\nLarge = {flops_max:.1f}B FLOPs', 
                transform=fig.transFigure, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make space for legend
        
        # Save the figure
        output_file = os.path.join(OUTPUT_DIR, 'multi_dimensional_model_comparison.png')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-dimensional comparison plot to {output_file}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating multi-dimensional comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())

def plot_metrics():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate and save comparative performance visualizations of different models
    training_metrics_file = os.path.join(RUN_DIR, "final_training_metrics.csv")
    performance_metrics_file = os.path.join(RUN_DIR, "final_performance_metrics.csv")
    
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
        inference_times = perf_metrics['inference_latency'].values
        params = training_metrics['trainable_params'].values
        flops = perf_metrics['flops'].values

        gflops = [round(float(flop / 1e9), 1) for flop in flops]  # Convert to GFLOPs with 1 decimal place
        
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
        
        # Calculate speed up
        speed_up = 20 * 60 / perf_metrics['inference_latency'].values
        logger.info(f"Generating performance comparisons for {len(model_names)} models")
        
        rmse_inverse = [1 - (r / max(rmse)) for r in rmse]  # Inverse RMSE for better visualization
        rmse_inverse_lable = 'Inverse RMSE (Accuracy)'
        # Fixed function calls without secondary axis parameters
        create_performance_plot(gflops, rmse_inverse, inference_times, model_names,
                              'GFLOPs', rmse_inverse_lable, 'Inference Time (s)',
                              'RMSE vs FLOPs vs Inference Latency', 'flops_vs_rmse_vs__inference_time.png')
        
        create_performance_plot(params,rmse_inverse, gflops, model_names,
                        'Parameters', rmse_inverse_lable, 'GFLOPs',
                        'RMSE vs FLOPs vs Inference Latency', 'parameters_vs_rmse_vs_flops.png')

        create_performance_plot(gflops, inference_memory_usage, params, model_names, 'GFLOPs', 
                                'Inference Memory Usage (GB)', 'Inference Time (s)',
                                'GFLOPs vs Inference Memory Usage vs Prameters', 
                                'flops_vs_inference_memory_vs_parameters.png')
        
        create_performance_plot(inference_memory_usage, rmse_inverse, inference_times, model_names,
                                'Inference Memory Usage (GB)', rmse_inverse_lable, 'Inference Time (s)',
                                'Inference Memory Usage vs RMSE vs Inference Latency', 
                                'inference_memory_vs_rmse_vs_inference_time.png')
        
        #compound footprint calculated based on the normalized values of flops, params and memory usage
        def normalize_metric(values):
            max_val = max(values)
            min_val = min(values)
            if max_val == min_val:
                return [0.5] * len(values)
            return [(val - min_val) / (max_val - min_val) for val in values]
        
        # Normalize each metric
        norm_gflops = normalize_metric(gflops)
        norm_params = normalize_metric(params)
        norm_memory = normalize_metric(inference_memory_usage)
        
        compound_footprint = []
        for i in range(len(model_names)):
            # Calculate compound footprint as weighted average of normalized metrics
            footprint = (norm_gflops[i] + norm_params[i] + norm_memory[i]) / 3
            compound_footprint.append(footprint)
        
        create_performance_plot(compound_footprint, rmse_inverse, inference_times, model_names, 'Compound Computational Footprint',
                                rmse_inverse_lable, 'Inference Time (s)',
                                'Compound Computational Footprint vs RMSE vs Inference Latency', 
                                'compound_footprint_vs_rmse_vs_inference_time.png')
        
        # create_performance_plot(params, inference_times, flops, model_names,
        #                       'Parameters (Million)', 'Inference Time (seconds)', 'FLOPs (Billion)',
        #                       'Model Complexity vs Inference Time', 'model_complexity_vs_inference_time.png')
        
        # create_performance_plot(flops, rmse, inference_times, model_names,
        #                       'FLOPs (Billion)', 'RMSE (m)', 'Inference Time (s)',
        #                       'Model Complexity and Latency vs Performance', 'flops_rmse_time.png')
        
        # create_performance_plot(inference_memory_usage, rmse, params, model_names,
        #                       'Inference Memory Usage (GB)', 'RMSE (m)', 'Parameters (Million)',
        #                       'Inference Memory Usage vs Parameters vs RMSE', 'inference_memory_usage_vs_rmse.png')
        
        # # Create the enhanced radar chart with all metrics
        # draw_radar_chart(model_names, rmse, inference_times, inference_memory_usage, flops, params)
            
        logger.info("Metrics visualizations generated successfully.")
    
    except Exception as e:
        logger.error(f"Error generating metrics visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    # Create comprehensive 6-subplot visualization in landscape orientation
    create_comprehensive_performance_subplot(model_names, gflops, rmse_inverse, inference_times, 
                                           params, inference_memory_usage, compound_footprint, 
                                           rmse_inverse_lable)
        
def create_comprehensive_performance_subplot(model_names, gflops, rmse_inverse, inference_times, 
                                           params, inference_memory_usage, compound_footprint, 
                                           rmse_inverse_lable):
    """Create a comprehensive 6-subplot performance visualization in landscape orientation."""
    
    # Create figure with 2x3 subplots in landscape orientation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    
    # Rename model_names to standard values
    display_names = []
    for model in model_names:
        if model == USRR_CNN1D_COMBINED:
            display_names.append("USRR-1DCNN")
        elif model == PICNN1D_V1:
            display_names.append("PI1DCNN")
        elif model == SRR_LSTM_COMBINED:
            display_names.append("SRR-LSTM")
        elif model == CNN1D_V1:
            display_names.append("1DCNN")
        elif model == HDL_FM_V1:
            display_names.append("HDL-FM")
        else:
            display_names.append(model)
    
    # Use consistent colors for all subplots
    colors = ['#0066CC'] * len(display_names)
    
    # Helper function to create each subplot
    def create_subplot(ax, x_vals, y_vals, z_vals, x_label, y_label, z_label, subplot_label):
        # Normalize z_values to bubble sizes
        def normalize_bubble_sizes(values, min_size=80, max_size=300):
            if max(values) == min(values):
                return [min_size] * len(values)
            normalized = [(val - min(values)) / (max(values) - min(values)) for val in values]
            return [min_size + (max_size - min_size) * norm for norm in normalized]
        
        bubble_sizes = normalize_bubble_sizes(z_vals)
        
        # Plot points
        for i, model in enumerate(display_names):
            ax.scatter(x_vals[i], y_vals[i], s=bubble_sizes[i], marker='o', 
                      color=colors[i], alpha=0.8, edgecolors='black', linewidth=1.5, label=model)
        
        # Add model name labels with arrows - black text
        for i, model in enumerate(display_names):
            ax.annotate(model, 
                      xy=(x_vals[i], y_vals[i]),
                      xytext=(8, 8),
                      textcoords='offset points',
                      fontsize=8,
                      fontweight='bold',
                      color='black',
                      bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="black", alpha=0.9),
                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black', lw=1.5))
        
        # Configure subplot
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='-', alpha=0.3, color='gray', linewidth=0.5)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_color('black')
        ax.tick_params(colors='black', which='both', labelsize=8)
        
        # Add subplot label and bubble size legend
        ax.set_title(subplot_label, fontsize=10, pad=10)
        ax.text(0.5, -0.15, f'Bubble Size = {z_label}', transform=ax.transAxes, 
               fontsize=8, fontstyle='italic', color='black', ha='center')
    
    # Plot (a): GFLOPs vs Accuracy (bubble = Inference Time)
    create_subplot(axes[0, 0], gflops, rmse_inverse, inference_times,
                  'GFLOPs', rmse_inverse_lable, 'Inference Time (s)', '(a) GFLOS vs Accuracy')
    
    # Plot (b): Parameters vs Accuracy (bubble = GFLOPs)
    create_subplot(axes[0, 1], params, rmse_inverse, gflops,
                  'Parameters', rmse_inverse_lable, 'GFLOPs', '(b) Parameters vs Accuracy')
        
    # Plot (c): GFLOPs vs Memory Usage (bubble = Parameters)
    create_subplot(axes[0, 2], params, inference_memory_usage, inference_times,
                  'Parameters', 'Inference Memory Usage (GB)', 'Inference Time (s)', '(c) Paramters vs Memory Usage')
    
        # # Plot (d): Parameters vs GFLOPs (bubble = Memory Usage)
    create_subplot(axes[1, 0], gflops, inference_memory_usage, inference_times,
                  'GFLOPs', 'Inference Memory Usage (GB)','Inference Time (s)', '(d) GLOPs vs Memory Usage')
    
    # Plot (e): Memory Usage vs Accuracy (bubble = Inference Time)
    create_subplot(axes[1, 1], inference_memory_usage, rmse_inverse, inference_times,
                  'Inference Memory Usage (GB)', rmse_inverse_lable, 'Inference Time (s)', '(e) Memory Usage vs Accuracy')
    
    # Plot (f): Compound Footprint vs Accuracy (bubble = Inference Time)
    create_subplot(axes[1, 2], compound_footprint, rmse_inverse, inference_times,
                  'Compound Computational Footprint', rmse_inverse_lable, 'Inference Time (s)', '(f) Compound Footprint vs Accuracy')

    
    # Remove individual legends and create a single legend
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_legend():
                ax.get_legend().remove()
    
    # Create single legend below all subplots
    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
    #           ncol=len(display_names), frameon=True, fancybox=True, shadow=True,
    #           title="Models", fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.25)  # Make space for legend and labels
    
    # Save the comprehensive plot
    output_file = os.path.join(OUTPUT_DIR, 'comprehensive_performance_comparison.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comprehensive performance comparison to {output_file}")
    plt.close()

