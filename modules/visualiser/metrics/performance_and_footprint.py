import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Make sure this import is at the module level
from modules.lib.constants import RUN_DIR, GRAPH_OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(GRAPH_OUTPUT_DIR, "perf_plots")


def create_performance_plot(x_values, y_values, model_names, x_label, y_label, title, filename, 
                           secondary_x=None, secondary_x_label=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert input values to numeric format if they aren't already
    numeric_x_values = []
    numeric_y_values = []
    
    for x_val, y_val in zip(x_values, y_values):
        # Convert x value if needed
        if isinstance(x_val, str):
            try:
                if x_val.startswith('{'):  # Dictionary as string
                    # Handle memory usage dictionaries
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
        
        numeric_x_values.append(numeric_x)
        numeric_y_values.append(numeric_y)
    
    # Use consistent marker shape but different colors for models
    colors = plt.cm.tab10(range(len(model_names)))
    marker_size = 200  # Fixed size for all markers
    
    # Plot each point with the same marker shape but different colors
    for i, model in enumerate(model_names):
        ax.scatter(
            numeric_x_values[i], 
            numeric_y_values[i],
            s=marker_size,  # Use fixed size
            marker='o',  # Use circles for all models
            color=colors[i],
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5,
            label=model
        )
    
    # Add secondary x-axis if provided
    has_secondary_axis = False
    if secondary_x is not None and secondary_x_label is not None:
        has_secondary_axis = True
        # Create secondary axis that shares the y-axis
        ax2 = ax.twiny()
        
        # Convert secondary x values if needed
        numeric_secondary_x = []
        for x_val in secondary_x:
            if isinstance(x_val, str):
                try:
                    numeric_secondary_x.append(float(x_val))
                except:
                    numeric_secondary_x.append(0)
            else:
                numeric_secondary_x.append(float(x_val) if x_val is not None else 0)
        
        # Plot points on the secondary axis (invisible, just for scale)
        for i, model in enumerate(model_names):
            ax2.scatter(
                numeric_secondary_x[i], 
                numeric_y_values[i],
                s=0,  # Make the points invisible
                alpha=0
            )
            
        # Set labels and format for secondary axis
        ax2.set_xlabel(secondary_x_label, fontsize=12, labelpad=10)  # Add padding to move label away from ticks
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(False)
        
        # Set limits for the secondary axis
        x2_min = min(numeric_secondary_x) - 0.05 * (max(numeric_secondary_x) - min(numeric_secondary_x))
        x2_max = max(numeric_secondary_x) + 0.05 * (max(numeric_secondary_x) - min(numeric_secondary_x))
        ax2.set_xlim(x2_min, x2_max)
    
    # Add quadrant lines and labels based on median values
    med_x = np.median(numeric_x_values)
    med_y = np.median(numeric_y_values)
    if has_secondary_axis:
        med_secondary_x = np.median(numeric_secondary_x)
    
    # Get current axis limits to ensure complete coverage
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Ensure the limits include all data points with some padding
    x_padding = 0.05 * (max(numeric_x_values) - min(numeric_x_values))
    y_padding = 0.05 * (max(numeric_y_values) - min(numeric_y_values))
    
    x_min = min(numeric_x_values) - x_padding
    y_min = min(numeric_y_values) - y_padding
    x_max = max(numeric_x_values) + x_padding
    y_max = max(numeric_y_values) + 3 * y_padding  # Extra padding at the top
    
    # Set these expanded limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add quadrant dividers
    ax.axvline(med_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(med_y, color='gray', linestyle='--', alpha=0.5)
    
    # Determine if "lower is better" for metrics
    y_lower_is_better = "RMSE" in y_label or "Time" in y_label or "Memory" in y_label
    x_lower_is_better = "FLOPs" in x_label or "Parameters" in x_label or "Neurons" in x_label or "Memory" in x_label
    # Define x_higher_is_better - speed is better when higher
    x_higher_is_better = "Speed" in x_label
    
    # For secondary x-axis, determine if lower is better
    if has_secondary_axis:
        secondary_x_lower_is_better = "Time" in secondary_x_label or "Memory" in secondary_x_label or "FLOPs" in secondary_x_label

    # Fill quadrants with appropriate colors based on what's "better"
    if y_lower_is_better:
        if not has_secondary_axis:
            # Original logic for just primary x-axis
            if x_higher_is_better:
                # Bottom right is best (high x, low y)
                best_quadrant = "bottom_right"
                worst_quadrant = "top_left"
            elif x_lower_is_better:
                # Bottom left is best (low x, low y)
                best_quadrant = "bottom_left"
                worst_quadrant = "top_right"
            else:
                # Bottom left is best by default for unlabeled x-axis (low x, low y)
                best_quadrant = "bottom_left"
                worst_quadrant = "top_right"
        else:
            # Both x-axis and secondary x-axis are considered
            if x_lower_is_better and secondary_x_lower_is_better:
                # Bottom left is best (low x, low secondary_x, low y)
                best_quadrant = "bottom_left"
                worst_quadrant = "top_right"
            else:
                # Default to bottom left is best
                best_quadrant = "bottom_left"
                worst_quadrant = "top_right"
    else:
        if x_higher_is_better:
            # Top right is best (high x, high y)
            best_quadrant = "top_right"
            worst_quadrant = "bottom_left"
        elif x_lower_is_better:
            # Top left is best (low x, high y)
            best_quadrant = "top_left"
            worst_quadrant = "bottom_right"
        else:
            # Top left is best by default for unlabeled x-axis (low x, high y)
            best_quadrant = "top_left"
            worst_quadrant = "bottom_right"
    
    # Add colored quadrants with more visible colors
    # Bottom left
    bl_color = "palegreen" if best_quadrant == "bottom_left" else ("lightcoral" if worst_quadrant == "bottom_left" else "khaki")
    ax.fill_between(
        [x_min, med_x], [y_min, y_min], [med_y, med_y], 
        color=bl_color, alpha=0.3  # Increased alpha for better visibility
    )
    
    # Bottom right
    br_color = "palegreen" if best_quadrant == "bottom_right" else ("lightcoral" if worst_quadrant == "bottom_right" else "khaki")
    ax.fill_between(
        [med_x, x_max], [y_min, y_min], [med_y, med_y], 
        color=br_color, alpha=0.3
    )
    
    # Top left
    tl_color = "palegreen" if best_quadrant == "top_left" else ("lightcoral" if worst_quadrant == "top_left" else "khaki")
    ax.fill_between(
        [x_min, med_x], [med_y, med_y], [y_max, y_max], 
        color=tl_color, alpha=0.3
    )
    
    # Top right
    tr_color = "palegreen" if best_quadrant == "top_right" else ("lightcoral" if worst_quadrant == "top_right" else "khaki")
    ax.fill_between(
        [med_x, x_max], [med_y, med_y], [y_max, y_max], 
        color=tr_color, alpha=0.3
    )
    
    # Create legend handles for quadrants
    from matplotlib.patches import Patch
    
    # Determine quadrant labels based on metrics
    if has_secondary_axis:
        # Custom labels for plots with 3 metrics
        if "FLOPs" in x_label and "Time" in secondary_x_label and "RMSE" in y_label:
            quadrant_handles = [
                Patch(facecolor="palegreen", edgecolor="green", alpha=0.7, 
                      label="BEST: Low FLOPs, Low Inference Time, Low Error"),
                Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, 
                      label="Mixed Performance"),
                Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, 
                      label="Mixed Performance"),
                Patch(facecolor="lightcoral", edgecolor="red", alpha=0.7, 
                      label="WORST: High FLOPs, High Inference Time, High Error")
            ]
        else:
            # Generic labels for other 3-metric combinations
            quadrant_handles = [
                Patch(facecolor="palegreen", edgecolor="green", alpha=0.7, 
                      label="BEST: Optimal Performance Region"),
                Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, 
                      label="Mixed Performance Region"),
                Patch(facecolor="khaki", edgecolor="gold", alpha=0.7, 
                      label="Mixed Performance Region"),
                Patch(facecolor="lightcoral", edgecolor="red", alpha=0.7, 
                      label="WORST: Poor Performance Region")
            ]
    else:
        # Original quadrant labels for 2-metric plots
        if y_lower_is_better:
            if x_higher_is_better:
                quadrant_handles = [
                    Patch(facecolor=bl_color, edgecolor="darkgreen" if best_quadrant == "bottom_left" else "gold", 
                          alpha=0.7, label="Low Speed, Low Error"),
                    Patch(facecolor=br_color, edgecolor="darkgreen" if best_quadrant == "bottom_right" else "gold", 
                          alpha=0.7, label="BEST: High Speed, Low Error" if best_quadrant == "bottom_right" else "High Speed, Low Error"),
                    Patch(facecolor=tl_color, edgecolor="darkgreen" if best_quadrant == "top_left" else "gold", 
                          alpha=0.7, label="Low Speed, High Error"),
                    Patch(facecolor=tr_color, edgecolor="darkgreen" if best_quadrant == "top_right" else "red", 
                          alpha=0.7, label="High Speed, High Error")
                ]
            else:
                quadrant_handles = [
                    Patch(facecolor=bl_color, edgecolor="darkgreen" if best_quadrant == "bottom_left" else "gold", 
                          alpha=0.7, label="BEST: Low Complexity, Low Error" if best_quadrant == "bottom_left" else "Low Complexity, Low Error"),
                    Patch(facecolor=br_color, edgecolor="darkgreen" if best_quadrant == "bottom_right" else "gold", 
                          alpha=0.7, label="High Complexity, Low Error"),
                    Patch(facecolor=tl_color, edgecolor="darkgreen" if best_quadrant == "top_left" else "gold", 
                          alpha=0.7, label="Low Complexity, High Error"),
                    Patch(facecolor=tr_color, edgecolor="darkgreen" if best_quadrant == "top_right" else "red", 
                          alpha=0.7, label="WORST: High Complexity, High Error")
                ]
        else:
            # Add this section to handle cases where higher y values are better
            if x_higher_is_better:
                quadrant_handles = [
                    Patch(facecolor=bl_color, edgecolor="darkgreen" if best_quadrant == "bottom_left" else "red", 
                          alpha=0.7, label="Low Speed, Low Performance"),
                    Patch(facecolor=br_color, edgecolor="darkgreen" if best_quadrant == "bottom_right" else "gold", 
                          alpha=0.7, label="High Speed, Low Performance"),
                    Patch(facecolor=tl_color, edgecolor="darkgreen" if best_quadrant == "top_left" else "gold", 
                          alpha=0.7, label="Low Speed, High Performance"),
                    Patch(facecolor=tr_color, edgecolor="darkgreen" if best_quadrant == "top_right" else "gold", 
                          alpha=0.7, label="BEST: High Speed, High Performance" if best_quadrant == "top_right" else "High Speed, High Performance")
                ]
            else:
                quadrant_handles = [
                    Patch(facecolor=bl_color, edgecolor="darkgreen" if best_quadrant == "bottom_left" else "red", 
                          alpha=0.7, label="Low Complexity, Low Performance"),
                    Patch(facecolor=br_color, edgecolor="darkgreen" if best_quadrant == "bottom_right" else "gold", 
                          alpha=0.7, label="High Complexity, Low Performance"),
                    Patch(facecolor=tl_color, edgecolor="darkgreen" if best_quadrant == "top_left" else "gold", 
                          alpha=0.7, label="BEST: Low Complexity, High Performance" if best_quadrant == "top_left" else "Low Complexity, High Performance"),
                    Patch(facecolor=tr_color, edgecolor="darkgreen" if best_quadrant == "top_right" else "gold", 
                          alpha=0.7, label="High Complexity, High Performance")
                ]
    
    # Add model name labels with arrows
    for i, model in enumerate(model_names):
        ax.annotate(model, 
                  xy=(numeric_x_values[i], numeric_y_values[i]),
                  xytext=(20, 20),
                  textcoords='offset points',
                  fontsize=12,
                  fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", fc=colors[i], ec="black", alpha=0.2),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='gray'))
    
    # Get handles and labels for the models
    handles, labels = ax.get_legend_handles_labels()
    
    # Create model legend
    first_legend = ax.legend(handles, labels, 
                           loc='upper center', bbox_to_anchor=(0.5, -0.10), 
                           ncol=len(model_names) if len(model_names) <= 4 else 4,
                           frameon=True, fancybox=True, shadow=True,
                           title="Models")
    
    # Add quadrant legend
    ax.add_artist(first_legend)
    if 'quadrant_handles' in locals():
        second_legend = ax.legend(handles=quadrant_handles, 
                                loc='upper center', bbox_to_anchor=(0.5, -0.25),
                                ncol=2, 
                                frameon=True, fancybox=True, shadow=True,
                                title="Quadrants")
    
    # Configure plot with consistent formatting
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    # Add grid and improve aesthetics
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure
    output_file = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.title(title, fontsize=12, pad=20)
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
        
        # Normalize values for radar chart
        def normalize(values):
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return [0.5] * len(values)  # All the same value
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        # For all these metrics, lower values are better, so invert them for the radar chart
        # Higher values on the radar chart = better performance
        norm_rmse = normalize([1/r for r in numeric_rmse])                 # Invert RMSE
        norm_inf_time = normalize([1/t for t in numeric_inference_times])  # Invert inference time
        norm_pred_memory = normalize([1/m for m in numeric_pred_memory])   # Invert memory
        norm_flops = normalize([1/f for f in numeric_flops])              # Invert FLOPs
        norm_params = normalize([1/p for p in numeric_params])            # Invert parameters
        
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
        
        # Create radar chart data with all metrics - use more descriptive labels
        categories = [
            'Model \nAccuracy',
            'Inference \nSpeed',
            'Memory \nEfficiency',
            'FLOPS \nEfficiency',
            'Model Parameters \nEfficiency'
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
        
    except Exception as e:
        logger.error(f"Error creating comprehensive radar chart: {e}")
        import traceback
        logger.error(traceback.format_exc())

def plot_metrics():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    """Generate and save comparative performance visualizations of different models."""
    training_metrics_file = os.path.join(RUN_DIR, "final_training_metrics.csv")
    performance_metrics_file = os.path.join(RUN_DIR, "final_performance_metrics.csv")
    
    try:
        train_metrics = pd.read_csv(training_metrics_file)
        training_metrics = train_metrics.groupby('model').last().reset_index()
        perf_metrics = pd.read_csv(performance_metrics_file)
        
        # Filter performance metrics to only include models in training metrics
        perf_metrics = perf_metrics[perf_metrics['run_id'].isin(training_metrics['run_id'])]
        model_names = perf_metrics['model'].values
        rmse = perf_metrics['rmse'].values
        inference_times = perf_metrics['inference_latency'].values
        params = training_metrics['trainable_params'].values
        flops = perf_metrics['flops'].values
        neurons = training_metrics['total_neurons'].values
        training_memory_usage = training_metrics['training_memory_usage'].values
        inference_memory_usage = perf_metrics['pred_memory_usage'].values
        
        #Add a new value for speed up
        speed_up = 20 * 60 /perf_metrics['inference_latency'].values
        logger.info(f"Generating performance comparisons for {len(model_names)} models")
        
        create_performance_plot(speed_up, rmse, model_names,
                                  'Speed Up (x)', 'RMSE (m)', 
                                  'Model Speed Up vs RMSE',
                                  'model_speed_up_vs_rmse.png')
        
        create_performance_plot(params, inference_times, model_names,
                                  'Parameters (Million)', 'Inference Time (seconds)',
                                  'Model Complexity vs Inference Time',
                                  'model_complexity_vs_inference_time.png')
        
        # Update the FLOPs vs RMSE plot to include inference time as marker size
        create_performance_plot(flops, rmse, model_names,
                              'FLOPs (Billion)', 'RMSE (m)',
                              'Model Complexity and Latency vs Performance',
                              'flops_rmse_time.png',
                              secondary_x=inference_times, 
                              secondary_x_label='Inference Time (seconds)')
        
        create_performance_plot(neurons, rmse, model_names,
                                  'Total Neurons (Million)', 'RMSE (m)',
                                  'Model Size vs RMSE',
                                  'model_size_vs_rmse.png')
        
        # Add parameters as secondary axis to the training memory usage vs RMSE plot
        create_performance_plot(inference_memory_usage, rmse, model_names,
                                'Inference Memory Usage (GB)', 'RMSE (m)',
                                'Inference Memory Usage vs Parameters vs RMSE',
                                'inference_memory_usage_vs_rmse.png',
                                secondary_x=params,
                                secondary_x_label='Parameters (Million)')
        
        draw_memory_usage_plot(training_memory_usage, inference_memory_usage, model_names, rmse)
        
        # Create the enhanced radar chart with all metrics
        draw_radar_chart(model_names, rmse, inference_times, inference_memory_usage, flops, params)
        
        logger.info("Metrics visualizations generated successfully.")
        
    except Exception as e:
        logger.error(f"Error generating metrics visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

