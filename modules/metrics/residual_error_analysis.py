import os
import pandas as pd
import logging
import torch
import glob
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from modules.lib.constants import RUN_DIR, SIMULATION_DATA_DIR, DATA_DIR, OUTPUT_DIR, CNN1D_V1, HDL_FM_V1, PICNN1D_V1, USRR_CNN1D_COMBINED
from modules.datamanager.datamanager import check_inundation_data_cache
from modules.metrics.bootstrap_analysis import mRMSE, bootstrap_function
from modules.lib.constants import  HITRATE, CSI, F2SCORE, F3SCORE, RMSE_T, RMSE_S, MRMSE_T
from modules.lib.constants import DEM_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name_map = {
    USRR_CNN1D_COMBINED: 'Tier-1',
    CNN1D_V1: 'Tier-2',
    PICNN1D_V1: 'Tier-3',
    HDL_FM_V1: 'Tier-4'
}
    
model_colors = {
    'Tier-1': "#0173B2",  
    'Tier-2': "#DE8F05",     
    'Tier-3': "#029E73",      
    'Tier-4': "#D55E00",      
}

# models = [USRR_CNN1D_COMBINED]
models = [USRR_CNN1D_COMBINED,CNN1D_V1,PICNN1D_V1, HDL_FM_V1]

def compute_and_plot_metrics():
    # residual_error_analysis()
    plot_flood_extent_maps()
    # rmse_vs_elevation_percentile()
    
def residual_error_analysis():
    reference_maps_tensor = preload_reference_inundation_maps()
    residual_metrics = {}
    for model_name in models:
        with torch.no_grad():
            torch.cuda.empty_cache()
            pred_maps_tensor  = load_output_maps(model_name)
            reference_maps_tensor_reshaped = reference_maps_tensor.view(reference_maps_tensor.shape[0], -1)
            #Create residual error map
            residual_maps = pred_maps_tensor - reference_maps_tensor_reshaped
            logger.info(f"residual maps shape: {residual_maps.shape}")
        
            # Calculate the RMSE for each time step
            rmse_per_timestep = torch.sqrt(torch.mean(residual_maps**2, dim=1))
            
            # RMSE per cell over time
            rmse_per_cell = torch.sqrt(torch.mean(residual_maps**2, dim=0))
            
            
            #Calculate mRMSE 
            overall_mrmse, mrmse_per_timestep = mRMSE(residual_maps, reference_maps_tensor_reshaped)
            
            #Calculate Hit rate, CSI, F2_Score and F3_Score for each timestep
            #calculate confusion matrix at 0.3m threshold
            threshold = 0.3
            pred_binary = torch.tensor(pred_maps_tensor > threshold).float()
            ref_binary = torch.tensor(reference_maps_tensor_reshaped > threshold).float()
            
            # Initialize arrays to store metrics for each timestep
            num_timesteps = pred_binary.shape[0]
            hit_rates = []
            csi_scores = []
            f2_scores = []
            f3_scores = []
            
            # Calculate confusion matrix and metrics for each timestep
            for t in range(num_timesteps):
                pred_t = pred_binary[t]
                ref_t = ref_binary[t]
                
                tp = ((pred_t == 1) & (ref_t == 1)).sum().item()
                tn = ((pred_t == 0) & (ref_t == 0)).sum().item()
                fp = ((pred_t == 1) & (ref_t == 0)).sum().item()
                fn = ((pred_t == 0) & (ref_t == 1)).sum().item()
                
                # Hit Rate (Sensitivity/Recall)
                hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                hit_rates.append(hit_rate)
                
                # Critical Success Index (CSI)
                csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                csi_scores.append(csi)
                
                # F2 Score
                f2_score = (tp - fn) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                f2_scores.append(f2_score)
                
                # F3 Score
                f3_score = (tp - fp) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                f3_scores.append(f3_score)
            
            # Convert to tensors for easier handling
            hit_rates = torch.tensor(hit_rates)
            csi_scores = torch.tensor(csi_scores)
            f2_scores = torch.tensor(f2_scores)
            f3_scores = torch.tensor(f3_scores)
            
            # Log overall statistics
            logger.info(f"Hit Rate - Mean: {hit_rates.mean():.3f}, Std: {hit_rates.std():.3f}, Max: {hit_rates.max():.3f}")
            logger.info(f"CSI - Mean: {csi_scores.mean():.3f}, Std: {csi_scores.std():.3f}, Max: {csi_scores.max():.3f}")
            logger.info(f"F2 Score - Mean: {f2_scores.mean():.3f}, Std: {f2_scores.std():.3f}")
            logger.info(f"F3 Score - Mean: {f3_scores.mean():.3f}, Std: {f3_scores.std():.3f}")
                
            residual_metrics[model_name] = {
                RMSE_T: rmse_per_timestep,
                RMSE_S: rmse_per_cell, 
                MRMSE_T: mrmse_per_timestep,
                HITRATE: hit_rates,
                CSI: csi_scores,
                F2SCORE: f2_scores,
                F3SCORE: f3_scores,
            }
            

            if model_name in [CNN1D_V1]:
                bootstrap_function(model_name_map.get(model_name), residual_maps, reference_maps_tensor_reshaped, num_samples=10000)
    
    # Plot the RMSE per timestep and RMSE per cell for all models
    plot_metrics(residual_metrics)
    
def plot_metrics(residual_metrics):
    # Temporal plots
    plot_temporal(residual_metrics, RMSE_T, r'$RMSE_{t}$ (m)', 'c) Temporal distribution of RMSE', 'rmse_per_timestep.png')
    plot_temporal(residual_metrics, MRMSE_T, r'$mRMSE_{t}$ (m)', 'c) Temporal distribution of masked RMSE (wet-cells only)', 'mrmse_per_timestep.png')
    
    # Boxplots
    plot_boxplot(residual_metrics, RMSE_T, r'$RMSE_{t}$ (m)', r'a) Distribution of $RMSE_{t}$', 'rmse_temporal_boxplot.png', 'rmse_temporal_stats_box_summary.csv')
    plot_boxplot(residual_metrics, MRMSE_T, r'$mRMSE_{t}$ (m)', r'b) Distribution of $mRMSE_{t}$', 'mrmse_temporal_boxplot.png', 'mrmse_temporal_stats_box_summary.csv')
    plot_boxplot(residual_metrics, RMSE_S, r'$RMSE_{s}$ (m)', r'Distribution of $RMSE_{s}$', 'rmse_per_cell_boxplot.png', 'rmse_spatial_stats_summary.csv')
    
    # Spatial maps
    plot_rmse_per_cell(residual_metrics)

def plot_temporal(residual_metrics, metric_key, ylabel, title, filename, add_peak_annotations=True):
    """
    Generic function to plot temporal evolution of any metric.
    
    Args:
        residual_metrics: Dictionary with model names as keys and metric dictionaries as values
        metric_key: Key to access the specific metric (e.g., RMSE_T, MRMSE_T)
        ylabel: Label for y-axis
        title: Plot title
        filename: Output filename
        add_peak_annotations: Whether to add peak value annotations
    """
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Get timesteps and convert to hours
    first_model = next(iter(residual_metrics.values()))
    timesteps = len(first_model[metric_key])
    hours = np.arange(0, timesteps * 0.25, 0.25)
    
    # Mark the peak of the event at 34.25 hours
    peak_hour = 34.25
    peak_timestep = int(peak_hour / 0.25)
    
    if peak_timestep < len(hours):
        peak_y_values = [residual_metrics[model][metric_key].cpu().numpy()[peak_timestep] 
                        for model in models if model in residual_metrics]
        max_peak_y = max(peak_y_values) if peak_y_values else ax1.get_ylim()[1]
        
        # Add vertical line at peak with annotation
        multiplier = 2.0 if 'mRMSE' in metric_key else 1.5
        ax1.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.7)
        ax1.annotate('Event Peak', xy=(peak_hour, max_peak_y * multiplier),
                    xytext=(peak_hour + 7, max_peak_y * multiplier),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                    fontsize=18, fontweight='bold')
    
    # Plot each model
    for model_key in models:
        if model_key not in residual_metrics:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        metric_values = residual_metrics[model_key][metric_key].cpu().numpy()
        ax1.plot(hours, metric_values, label=model_display_name, color=model_colors[model_display_name], linewidth=2)
        
        if add_peak_annotations:
            # Annotate peak point for each model
            peak_timestep_idx = np.argmax(metric_values)
            peak_hour_model = hours[peak_timestep_idx]
            peak_value = metric_values[peak_timestep_idx]
            
            # Model-specific offsets
            x_offset, y_offset = get_annotation_offsets(model_key, metric_key)
            
            if peak_value is not None:
                ax1.annotate(f'{peak_value:.2f} m', xy=(peak_hour_model, peak_value),
                             xytext=(peak_hour_model + x_offset, peak_value + y_offset),
                             arrowprops=dict(facecolor=model_colors[model_display_name], shrink=0.05, width=1.5, headwidth=7),
                             fontsize=18 if metric_key == RMSE_T else 14, color=model_colors[model_display_name],
                             bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=model_colors[model_display_name], alpha=0.9))
    
    ax1.set_xlabel('Time (hours)', fontsize=18)
    ax1.set_ylabel(ylabel, fontsize=18)
    ax1.set_title(title, fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=18)
    
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', filename)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {metric_key} temporal plot at {outfile}")


def get_annotation_offsets(model_key, metric_key):
    """Helper function to get annotation offsets for different models and metrics."""
    offsets = {
        RMSE_T: {
            HDL_FM_V1: (3, -0.03),
            PICNN1D_V1: (2, 0.00),
            USRR_CNN1D_COMBINED: (3, -0.03),
            CNN1D_V1: (3, -0.02)
        },
        MRMSE_T: {
            HDL_FM_V1: (2, -0.1),
            PICNN1D_V1: (5, 0.2),
            USRR_CNN1D_COMBINED: (5, -0.1),
            CNN1D_V1: (3, 0.1)
        }
    }
    return offsets.get(metric_key, {}).get(model_key, (5, 0.1))


def plot_boxplot(residual_metrics, metric_key, ylabel, title, filename, stats_filename, show_median=False):
    """
    Generic function to create boxplots for any metric.
    
    Args:
        residual_metrics: Dictionary with model names as keys and metric dictionaries as values
        metric_key: Key to access the specific metric
        ylabel: Label for y-axis
        title: Plot title
        filename: Output filename
        stats_filename: CSV filename for statistics
        show_median: Whether to show median separately (for spatial metrics)
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    box_data = []
    display_names = []
    
    for model_key in models:
        if model_key not in residual_metrics:
            continue
            
        metric_values = residual_metrics[model_key][metric_key].cpu().numpy()
        
        # Filter out zeros for spatial metrics, keep all for temporal
        if metric_key in [RMSE_S]:
            filtered_values = metric_values[metric_values > 0]
        else:
            filtered_values = metric_values
            
        box_data.append(filtered_values)
        display_names.append(model_name_map.get(model_key, model_key))
    
    # Create boxplot
    bp = plt.boxplot(box_data, patch_artist=True, notch=True, showfliers=True, 
                    flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markersize': 6 if metric_key in [RMSE_S] else 6, 'linestyle': 'none'})
    
    # Customize colors
    for i, box in enumerate(bp['boxes']):
        model_key = models[i]
        model_display_name = model_name_map.get(model_key, model_key)
        box.set(color=model_colors[model_display_name], linewidth=2)
        box.set(facecolor=model_colors[model_display_name], alpha=0.7)
    
    for element in ['whiskers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set(color='black', linewidth=2)
    
    for i, fliers in enumerate(bp['fliers']):
        model_key = models[i]
        model_display_name = model_name_map.get(model_key, model_key)
        fliers.set(marker='o', markerfacecolor=model_colors[model_display_name],
                  markeredgecolor='gray', markersize=4 if metric_key in [RMSE_S] else 6, 
                  alpha=0.5, linestyle='none')
    
    # Add statistics
    stats_csv_file = os.path.join(OUTPUT_DIR, 'quality_metrics', stats_filename)
    stats_df = pd.read_csv(stats_csv_file) if os.path.exists(stats_csv_file) else pd.DataFrame()
    
    positions = range(1, len(display_names) + 1)
    
    for i, data in enumerate(box_data):
        pos = positions[i]
        model_key = models[i]
        model_display_name = display_names[i]
        
        # Calculate statistics
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        median = np.median(data)
        mean = np.mean(data)
        # Show annotations based on metric type
        if show_median:
            # For spatial metrics with median shown separately
            plt.text(pos - 0.3, median, f'Med: {median:.3f}', ha='right', va='center', 
                    fontsize=18, rotation=90, color=model_colors[model_display_name],
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=model_colors[model_display_name], alpha=0.7))
            plt.text(pos + 0.3, (q1 + q3)/2, f'IQR: {iqr:.3f}', ha='left', va='center', 
                    fontsize=18, rotation=90, color=model_colors[model_display_name],
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=model_colors[model_display_name], alpha=0.7))
        else:
            # For temporal metrics with combined annotation
            y_pos = (q1 + q3)/2
            if i == 3 and (metric_key == MRMSE_T or metric_key == RMSE_S):  # Special case adjustment
                y_pos += 0.2
            plt.text(pos + 0.3, y_pos, f'Med: {median:.2f} IQR: {iqr:.2f}', ha='left', va='center', 
                    fontsize=15, rotation=90, color=model_colors[model_display_name],
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=model_colors[model_display_name], alpha=0.7))
        
        # Save statistics
        new_row = {
            'Model': model_display_name,
            'Mean_RMSE_m': np.mean(data),
            'Max_RMSE_m': np.max(data),
            'Min_RMSE_m': np.min(data[data > 0]) if np.any(data > 0) else np.min(data),
            'Median_RMSE_m': median,
            'IQR_RMSE_m': iqr
        }
        stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)
    
    stats_df.to_csv(stats_csv_file, index=False)
    
    plt.xticks(range(1, len(display_names) + 1), display_names, fontsize=18, rotation=0)
    plt.yticks(fontsize=15 if metric_key in [RMSE_T, MRMSE_T] else 18)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
    
    os.makedirs(os.path.join(OUTPUT_DIR, 'quality_metrics'), exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', filename)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {metric_key} boxplot at {outfile}")

def plot_rmse_per_cell(residual_metrics):
    # Plot the spatial distribution of RMSE per cell for each model using DEM
    out_path = os.path.join(OUTPUT_DIR, 'quality_metrics', 'rmse_per_cell_maps')
    os.makedirs(out_path, exist_ok=True)
    
    # Read DEM for background context
    try:
        with rio.open(DEM_FILE) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            height, width = dem.shape
            dem_nodata = src.nodata  # Get no data value for proper masking
            logger.info(f"DEM dimensions: {height}x{width}")
    except Exception as e:
        logger.error(f"Could not open DEM file {DEM_FILE}: {e}")
        dem = None
        dem_nodata = None
    
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # Plot each model's RMSE spatial distribution
    for model_key in ordered_models:
        if model_key not in residual_metrics:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        logger.info(f"Plotting RMSE map for {model_display_name}")
        
        # Get RMSE per cell and reshape to 2D
        rmse_per_cell = residual_metrics[model_key][RMSE_S].cpu().numpy()
        
        try:
            # Reshape from flattened to 2D grid
            rmse_grid = rmse_per_cell.reshape(611, 951)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot DEM as background if available (match plot_flood_extent_maps)
            # if dem is not None:
            #     # Create a masked version of DEM for better visualization
            #     dem_masked = np.ma.masked_equal(dem, dem_nodata)  # Consistent mask with plot_flood_extent_maps
            #     # Using origin='upper' ensures correct orientation (north at top)
            #     # Don't use extent with the DEM as it can cause scaling issues
            #     ax.imshow(dem_masked, cmap='gray', alpha=0.3, origin='upper')
            
            # Create colormap for RMSE using a 6-bin discrete scheme
            from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
            # Define RMSE bins and corresponding colors
            # Bins: 0, (0, 0.15], (0.15, 0.3], (0.3, 0.5], (0.5, 1.0], (1.0, inf)
            boundaries = [0, 0.1, 0.3, 0.5,100]  # Use large value for upper bound
            colors_list = [
                (1.0, 1.0, 1.0),      # White for RMSE = 0
                (0.5, 0.5, 0.5),      # Grey for 0 < RMSE <= 0.15
                (1.0, 1.0, 0.0),      # Yellow for 0.15 < RMSE <= 0.3
                (1.0, 0.55, 0.0),     # Orange for 0.3 < RMSE <= 0.5
                (1.0, 0.0, 0.0),      # Red for 0.5 < RMSE <= 1.0
            ]
            cmap = LinearSegmentedColormap.from_list('rmse_discrete', colors_list, N=len(colors_list))
            norm = BoundaryNorm(boundaries, cmap.N, clip=True)
            
            # Calculate global min and max for consistent scale across all plots
            # Collect all RMSE values for setting consistent scale
            all_rmse_values = []
            for model_key_inner in ordered_models:
                if model_key_inner in residual_metrics:
                    all_rmse_values.append(residual_metrics[model_key_inner][RMSE_S].cpu().numpy())
            
            # Combine all values and calculate global max for scaling (exclude zeros)
            all_rmse = np.concatenate([v[v > 0] for v in all_rmse_values])
            vmin = 0
            vmax = np.percentile(all_rmse, 95)  # Use 95th percentile to avoid extreme outliers
            
            # Plot RMSE map with transparency for zeros
            # Use the discrete colormap with normalization
            im = ax.imshow(rmse_grid, cmap=cmap, norm=norm, alpha=0.7, origin='upper')
            
            # Add title and labels
            ax.set_title(f'{model_display_name} Model', fontsize=28)
            
            # Define geographic extent
            west = transform[0]
            north = transform[3]
            pixel_width = transform[1]
            pixel_height = abs(transform[5])  # Usually negative, so take absolute
            
            # Calculate extent corners in real-world coordinates
            east = west + width * pixel_width
            south = north - height * pixel_height
            extent = [west, east, south, north]
            
            # Create tick positions - show approximately 5 ticks on each axis
            x_step = int(width / 5)
            y_step = int(height / 5)
            
            x_ticks = np.arange(0, width, x_step)
            y_ticks = np.arange(0, height, y_step)
            
            # Convert tick positions to real-world coordinates
            x_tick_labels = [f'{int(west + x * pixel_width)}' for x in x_ticks]
            y_tick_labels = [f'{int(north - y * pixel_height)}' for y in y_ticks]
            
            # Add scale bar and north arrow (matching plot_flood_extent_maps)
            ax.text(0.95, 0.05, '↑N', transform=ax.transAxes, fontsize=12, 
                    fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
            
            # Add scale bar - use pixel coordinates for now since geographic doesn't seem to work
            if dem is not None:
                # Use pixel coordinates instead of geographic coordinates
                scale_x_pix = width * 0.05  # 5% from left
                scale_y_pix = height * 0.95  # 95% from top
                scalebar_length_pix = width * 0.1  # 10% of width
                
                # Draw the scale bar in pixel coordinates
                ax.plot([scale_x_pix, scale_x_pix + scalebar_length_pix], [scale_y_pix, scale_y_pix], 'k-', linewidth=2)
                ax.text(scale_x_pix + scalebar_length_pix/2, scale_y_pix - height * 0.01, 
                       f'500m', ha='center', va='bottom', 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            
            # Remove axis ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add text with statistics in a box similar to plot_flood_extent_maps
            stats_text = f"Mean RMSE: {np.mean(rmse_per_cell):.3f} m"
            
            # Calculate percentages for each color bin
            high_rmse_percentage = (np.sum(rmse_per_cell > 0.5) / np.sum(rmse_per_cell > 0)) * 100
            mid_rmse_percentage = (np.sum((rmse_per_cell > 0.3) & (rmse_per_cell <= 0.5)) / np.sum(rmse_per_cell > 0)) * 100
            lower_rmse_percentage = (np.sum((rmse_per_cell > 0.1) & (rmse_per_cell <= 0.3)) / np.sum(rmse_per_cell > 0)) * 100
            grey_rmse_percentage = (np.sum((rmse_per_cell > 0) & (rmse_per_cell <= 0.1)) / np.sum(rmse_per_cell > 0)) * 100
            
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=24,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # Create a visual legend with colored rectangles at bottom right
            from matplotlib.patches import Rectangle
            
            # Define legend position and size
            legend_x = 0.88
            legend_y = 0.02
            box_width = 0.03
            box_height = 0.04
            spacing = 0.05
            
            # Define colors matching the colormap
            legend_colors = [
                (0.5, 0.5, 0.5),      # Grey
                (1.0, 1.0, 0.0),      # Yellow
                (1.0, 0.55, 0.0),     # Orange
                (1.0, 0.0, 0.0),      # Red
            ]
            
            legend_labels = [
                f'0-0.1m ({grey_rmse_percentage:.1f}%)',
                f'0.1-0.3m ({lower_rmse_percentage:.1f}%)',
                f'0.3-0.5m ({mid_rmse_percentage:.1f}%)',
                f'>0.5m ({high_rmse_percentage:.1f}%)'
            ]
            
            # Draw colored boxes and labels from bottom to top
            for i, (color, label) in enumerate(zip(legend_colors, legend_labels)):
                y_pos = legend_y + i * spacing
                
                # Draw colored rectangle
                rect = Rectangle((legend_x - box_width, y_pos), box_width, box_height,
                                transform=ax.transAxes, facecolor=color, 
                                edgecolor='black', linewidth=1, clip_on=False)
                ax.add_patch(rect)
                
                # Add label text to the left of the box
                ax.text(legend_x - box_width - 0.01, y_pos + box_height/2, label,
                       transform=ax.transAxes, fontsize=14, 
                       verticalalignment='center', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', 
                                alpha=0.9, edgecolor='gray'))
            
            #save the min, max, median, iqr to rmse_stats_df
            stats_csv_file = os.path.join(out_path, 'rmse_per_cell_summary.csv')
            if os.path.exists(stats_csv_file):
                rmse_stats_df = pd.read_csv(stats_csv_file)
            else:
                rmse_stats_df = pd.DataFrame()  
        
            new_row = {
                'Model': model_display_name,
                'Mean_RMSE_m': np.mean(rmse_per_cell[rmse_per_cell > 0]),
                'Max_RMSE_m': np.max(rmse_per_cell),
                'Min_RMSE_m': np.min(rmse_per_cell[rmse_per_cell > 0]), 
                'Median_RMSE_m': np.median(rmse_per_cell[rmse_per_cell > 0]),
                'IQR_RMSE_m': np.percentile(rmse_per_cell[rmse_per_cell > 0], 75) - np.percentile(rmse_per_cell[rmse_per_cell > 0], 25)
            }
            
            rmse_stats_df = pd.concat([rmse_stats_df, pd.DataFrame([new_row])], ignore_index=True)
            rmse_stats_df.to_csv(stats_csv_file, index=False)       
            
            # Save figure
            outfile = os.path.join(out_path, f'{model_display_name}_rmse_map.png')
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved RMSE map for {model_display_name} at {outfile}")
            
        except Exception as e:
            logger.error(f"Error creating RMSE map for {model_display_name}: {e}")
            continue
    
    # Create a separate colorbar image
    try:
        logger.info("Creating separate colorbar image for RMSE maps")
        # Create a separate figure for the colorbar
        fig_cb = plt.figure(figsize=(9, 1.5))
        ax_cb = fig_cb.add_axes([0.1, 0.3, 0.8, 0.4])
        
        # Create the same discrete colormap as used in the RMSE maps
        from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
        boundaries = [0, 0.10, 0.3, 0.5, 100]
        colors_list = [
            (1.0, 1.0, 1.0),      # White
            (0.5, 0.5, 0.5),      # Grey
            (1.0, 1.0, 0.0),      # Yellow
            (1.0, 0.55, 0.0),     # Orange
            (1.0, 0.0, 0.0),      # Red
        ]
        cmap = LinearSegmentedColormap.from_list('rmse_discrete', colors_list, N=len(colors_list))
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)
        
        # Create a horizontal colorbar with discrete bins
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax_cb, orientation='horizontal', 
                        boundaries=boundaries, ticks=[0.075, 0.225, 0.4, 0.75, 1.5])
        cb.ax.set_xticklabels(['0-0.10', '0.15-0.3', '0.3-0.5', '>0.5'])
        cb.set_label(r'RMSE (m)', fontsize=12, fontweight='bold')
        cb.ax.tick_params(labelsize=10)
        
        # Save the colorbar image
        colorbar_file = os.path.join(out_path, "rmse_colorbar.png")
        plt.savefig(colorbar_file, dpi=300, bbox_inches='tight')
        plt.close(fig_cb)
        logger.info(f"Saved separate RMSE colorbar image to {colorbar_file}")
    except Exception as e:
        logger.error(f"Error creating separate colorbar image: {e}")
        
    logger.info(f"Completed generating RMSE maps for all available models")

def plot_flood_extent_maps():
    """
    Creates confusion matrix maps for flood extent predictions showing:
    A: Hits (TP) - correctly predicted flooded areas
    B: Overpredictions (FP) - predicted flood where none exists
    C: Misses (FN) - missed flooded areas (underpredictions)
    D: True Negatives (TN) - correctly predicted dry areas
    
    Uses a threshold of 0.3m to classify flooded vs dry areas.
    """
    logger.info("Creating flood extent confusion matrix maps")
    
    # List of all models to process
    model_names = [USRR_CNN1D_COMBINED, CNN1D_V1, PICNN1D_V1, HDL_FM_V1]
    
    # Create nicer display names for models
    model_display_names = {
        CNN1D_V1: 'Tier-2',
        PICNN1D_V1: 'Tier-3',
        USRR_CNN1D_COMBINED: 'Tier-1',
        HDL_FM_V1: 'Tier-4',
    }
    
    # Timestep to use for comparison
    reference_indices = ["0145", "0056", "0224"]  # Prioritized list of timesteps
    pred_idx = ["0137", "0048", "0216"]  # Alternative timesteps if primary isn't available
    
    for ref_idx, pred_idx in zip(reference_indices, pred_idx):
        # Define the flood threshold
        flood_threshold = 0.3
        
        # Reference LISFLOOD run (ground truth)
        lf_extent_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{ref_idx}.wd")
        if not os.path.exists(lf_extent_file):
            logger.error(f"Ground truth file doesn't exist: {lf_extent_file}")
            return None
        
        # Load the reference LISFLOOD data
        with rasterio.open(lf_extent_file) as src:
            truth_data = src.read(1)
            truth_nodata = src.nodata
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Load DEM data for background
        with rasterio.open(DEM_FILE) as src:
            dem_data = src.read(1)
            dem_nodata = src.nodata
        
        # Create binary truth mask (1 = flooded, 0 = dry)
        truth_binary = np.where(truth_data >= flood_threshold, 1, 0)
        
        # Define output directory
        output_dir = os.path.join(OUTPUT_DIR, "quality_metrics", "confusion_matrix_maps")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define colors for confusion matrix categories
        confusion_colors = {
            0: "#ffffff",  # True Negatives (TN) - White (correct dry)
            1: "#808080",  # True Positives (TP) - Grey (hits - correct flood)
            2: "#FF8C00",  # False Positives (FP) - Dark orange (overpredictions)
            3: "#FF0000",  # False Negatives (FN) - Red (misses - underpredictions)
        }
        
        # Create custom colormap
        from matplotlib.colors import ListedColormap
        colors_list = [confusion_colors[i] for i in range(4)]
        confusion_cmap = ListedColormap(colors_list)
        
        # Process each model
        available_models = []
        model_stats = {}
        
        for model_name in model_names:
            # Try to find the prediction file
            maps_dir = os.path.join(RUN_DIR, "output_maps", model_name)
            primary_map_path = os.path.join(maps_dir, f"map_{pred_idx}.wd")
            # alt_map_path = os.path.join(maps_dir, f"map_{pred_idx}.wd")
            
            model_map_path = None
            if os.path.exists(primary_map_path):
                model_map_path = primary_map_path

            if not model_map_path:
                logger.warning(f"No prediction data found for {model_name}")
                continue
            
            try:
                # Load the prediction data
                with rasterio.open(model_map_path) as src:
                    pred_data = src.read(1)
                    pred_nodata = src.nodata
                
                # Create binary prediction mask (1 = flooded, 0 = dry)
                pred_binary = np.where(pred_data >= flood_threshold, 1, 0)
                
                # Create confusion matrix map
                # 0: TN (both dry), 1: TP (both flooded), 2: FP (pred flood, truth dry), 3: FN (pred dry, truth flood)
                confusion_map = np.zeros_like(truth_binary)
                
                # True Negatives (TN) - both predict and truth are dry
                tn_mask = (pred_binary == 0) & (truth_binary == 0)
                confusion_map[tn_mask] = 0
                
                # True Positives (TP) - both predict and truth are flooded
                tp_mask = (pred_binary == 1) & (truth_binary == 1)
                confusion_map[tp_mask] = 1
                
                # False Positives (FP) - predict flooded but truth is dry (overpredictions)
                fp_mask = (pred_binary == 1) & (truth_binary == 0)
                confusion_map[fp_mask] = 2
                
                # False Negatives (FN) - predict dry but truth is flooded (misses/underpredictions)
                fn_mask = (pred_binary == 0) & (truth_binary == 1)
                confusion_map[fn_mask] = 3
                
                # Calculate confusion matrix statistics
                tp_count = np.sum(tp_mask)
                tn_count = np.sum(tn_mask)
                fp_count = np.sum(fp_mask)
                fn_count = np.sum(fn_mask)
                
                # Calculate metrics
                hit_rate = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
                csi = tp_count / (tp_count + fn_count + fp_count) if (tp_count + fn_count + fp_count) > 0 else 0
                f2_score = (tp_count - fn_count)/ (tp_count + fn_count + fp_count) if (tp_count + fn_count + fp_count) > 0 else 0
                f3_score  = (tp_count - fp_count) / (tp_count + fn_count + fp_count) if (tp_count + fn_count + fp_count) > 0 else 0
                
                # Store statistics
                display_name = model_display_names.get(model_name, model_name)
                model_stats[display_name] = {
                    'tp': tp_count, 'tn': tn_count, 'fp': fp_count, 'fn': fn_count,
                    'hit_rate': hit_rate, 'csi': csi, 'f2_score': f2_score, 'f3_score': f3_score
                }
                
                # Create the visualization
                fig, ax = plt.subplots(figsize=(16, 12))
                
                # Plot DEM as background (very light)
                dem_masked = np.ma.masked_equal(dem_data, dem_nodata)
                ax.imshow(dem_masked, extent=extent, cmap='terrain', alpha=0.8, origin='upper')
                
                # Plot confusion matrix map
                im = ax.imshow(confusion_map, extent=extent, cmap=confusion_cmap, 
                            alpha=0.8, origin='upper', vmin=0, vmax=3)
        
                
                # Set title with statistics
                # {chr(97 + len(available_models))}) 
                ax.set_title(f'{display_name} - Timestep {int(pred_idx)/4:.2f}h', 
                            fontsize=50, pad=15)
            
                # f'Hit-rate: {hit_rate:.2f}, CSI: {csi:.2f}, F2: {f2_score:.2f}, F3: {f3_score:.2f}'
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add border to the subplot
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.0)
                    spine.set_edgecolor('black')
                
                # Add scale bar and north arrow
                ax.text(0.95, 0.05, '↑N', transform=ax.transAxes, fontsize=12, 
                    fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
                
                scalebar_length_m = 500
                scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
                scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
                ax.plot([scale_x, scale_x + scalebar_length_m], [scale_y, scale_y], 'k-', linewidth=2)
                ax.text(scale_x + scalebar_length_m/2, scale_y + (extent[3] - extent[2]) * 0.01, 
                    f'{scalebar_length_m}m', ha='center', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
                
                # Add statistics text box
                stats_text = (f"A: {tp_count:,}\n"
                            f"B: {fp_count:,}\n"
                            f"C: {fn_count:,}\n"
                            f"D: {tn_count:,}")
                
                ax.text(0.98,0.02,stats_text, transform=ax.transAxes, fontsize=40,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
                
                plt.tight_layout()
                
                # Save individual model confusion matrix map
                output_file = os.path.join(output_dir, f"{display_name}_confusion_matrix{pred_idx}.png")
                plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Saved {display_name} confusion matrix map to {output_file}")
                plt.close()
                
                available_models.append(display_name)
                
            except Exception as e:
                logger.error(f"Error processing {model_name}: {str(e)}")
                continue
        
        if not available_models:
            logger.error("No model prediction data could be loaded")
            return None
        
        # Create legend figure
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        ax.axis('off')
        
        # Create legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=confusion_colors[0], edgecolor='black', 
                label='True Negatives (Correct Dry)'),
            Patch(facecolor=confusion_colors[1], edgecolor='black', 
                label='True Positives (Hits)'),
            Patch(facecolor=confusion_colors[2], edgecolor='black', 
                label='False Positives (Overpredictions)'),
            Patch(facecolor=confusion_colors[3], edgecolor='black', 
                label='False Negatives (Misses/Underpredictions)')
        ]
        
        ax.legend(handles=legend_elements, loc='center', fontsize=18, 
                title=f"Flood Extent Classification (Threshold: {flood_threshold}m)",
                title_fontsize=20, frameon=True, fancybox=True, shadow=True)
        

        
        plt.tight_layout()
        legend_file = os.path.join(output_dir, "confusion_matrix_legend.png")
        plt.savefig(legend_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved confusion matrix legend to {legend_file}")
        plt.close()
        
        # Create a separate legend figure for A,B,C,D labels
        abc_text = ("A: Hits (TP) - Correctly predicted flooded areas\n"
                    "B: Overpredictions (FP) - Predicted flood where none exists\n"
                    "C: Misses (FN) - Missed flooded areas (underpredictions)\n"
                    "D: True Negatives (TN) - Correctly predicted dry areas")
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('off')
        ax.text(0.5, 0.5, abc_text, transform=ax.transAxes, fontsize=18,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))    
        
        plt.tight_layout()
        abc_legend_file = os.path.join(output_dir, "confusion_matrix_ABC_legend.png")
        plt.savefig(abc_legend_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved confusion matrix A,B,C,D legend to {abc_legend_file}")
        plt.close()
        
        # Create summary statistics table and save as image
        create_confusion_matrix_summary(model_stats, output_dir, ref_idx)
        
        logger.info(f"Generated confusion matrix maps for {len(available_models)} models")
        logger.info("Summary statistics:")
        
    return output_dir
    
def rmse_vs_elevation_percentile():
    """
    Analyzes and plots model prediction accuracy across different DEM elevation percentiles.
    Creates a grid of plots showing error metrics for 6 different elevation ranges.
    """
    
    logger.info("Generating elevation-based prediction accuracy plots for all models")
    model_names = models
    
    # Create output directory for plots
    output_path = os.path.join(OUTPUT_DIR, "quality_metrics")
    os.makedirs(output_path, exist_ok=True)
    
    # Load DEM file
    try:
        with rio.open(DEM_FILE) as src:
            dem = src.read(1)
            transform = src.transform
            height, width = dem.shape
            dem_nodata = src.nodata
            logger.info(f"Loaded DEM with dimensions {height}x{width}")
    except Exception as e:
        logger.error(f"Could not open DEM file {DEM_FILE}: {e}")
        return
    
    # Mask out nodata values
    dem_valid = dem.copy()
    if dem_nodata is not None:
        dem_valid = np.ma.masked_equal(dem, dem_nodata)
        
    # Calculate elevation percentiles
    dem_flat = dem_valid.compressed()  # Remove masked values
    percentiles = [5, 10, 25, 40, 60]  # Added 1st percentile for finer low elevation analysis
    elev_thresholds = np.percentile(dem_flat, percentiles)
    
    logger.info(f"Elevation percentiles: {percentiles}")
    logger.info(f"Corresponding elevation thresholds: {elev_thresholds}")
    
    # Create elevation range bins
    elev_ranges = [(float('-inf'), elev_thresholds[0])]
    for i in range(len(elev_thresholds)-1):
        elev_ranges.append((elev_thresholds[i], elev_thresholds[i+1]))
    elev_ranges.append((elev_thresholds[-1], float('inf')))
    
    # Create range labels
    range_labels = [
        f"< {elev_thresholds[0]:.1f}m (< 5th %ile)",
        f"{elev_thresholds[0]:.1f}m-{elev_thresholds[1]:.1f}m (5th-10th)",
        f"{elev_thresholds[1]:.1f}m-{elev_thresholds[2]:.1f}m (10th-25th)",
        f"{elev_thresholds[2]:.1f}m-{elev_thresholds[3]:.1f}m (25th-40th)",
        f"{elev_thresholds[3]:.1f}m-{elev_thresholds[4]:.1f}m (40th-60th)",
        f">= {elev_thresholds[4]:.1f}m (>= 60th %ile)"
    ]
    
    # Plot the elevation ranges in different colors over the DEM for verification
    plt.figure(figsize=(12, 10))
    
    # Create a combined mask array where each cell has a value corresponding to its elevation range
    elevation_range_map = np.zeros_like(dem_valid.data, dtype=float)
    elevation_range_map[:] = np.nan  # Initialize with NaN for nodata areas
    
    # Assign each cell to its elevation range (0-5)
    for i, (elev_min, elev_max) in enumerate(elev_ranges):
        if elev_min == float('-inf'):
            mask = (dem_valid <= elev_max)
        elif elev_max == float('inf'):
            mask = (dem_valid > elev_min)
        else:
            mask = (dem_valid > elev_min) & (dem_valid <= elev_max)
        elevation_range_map[mask] = i
    
    # Create discrete colormap with distinct colors for each range
    colors_discrete = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde724', '#FFA500']  # Viridis-like + orange
    cmap_discrete = plt.matplotlib.colors.ListedColormap(colors_discrete[:len(elev_ranges)])
    bounds = np.arange(len(elev_ranges) + 1) - 0.5
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap_discrete.N)
    
    # Plot the elevation range map
    im = plt.imshow(elevation_range_map, cmap=cmap_discrete, norm=norm, origin='upper')
    
    # Create a custom legend instead of colorbar
    from matplotlib.patches import Patch
    legend_elements = []
    for i, label in enumerate(range_labels):
        legend_elements.append(Patch(facecolor=colors_discrete[i], edgecolor='black', label=label))
    
    # Add legend to the plot at the bottom with two rows
    ax = plt.gca()
    legend = ax.legend(handles=legend_elements, 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, -0.02),
                      ncol=3,  # 3 columns to get 2 rows with 6 items
                    #   title='Elevation Range',
                      title_fontsize=14,
                      fontsize=12,
                      frameon=True,
                      fancybox=True)
    
    # Make the legend title bold
    legend.get_title().set_fontweight('bold')
    
    # plt.title("Spatial distribution of elevation percentiles in the modelling domain", fontsize=18, pad=10)
    
    # Remove axis ticks and labels - the colorbar serves as the legend
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    
    # Remove the axis frame for a cleaner look
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    
    # Add north arrow
    plt.text(0.95, 0.05, '↑N', transform=plt.gca().transAxes, fontsize=14, 
            fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "dem_elevation_ranges.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved DEM elevation ranges overlay plot. to {os.path.join(output_path, 'dem_elevation_ranges.png')}")
    
    # Dictionary to store model data and results
    model_data = {}
    
    # Get reference maps
    reference_maps = preload_reference_inundation_maps()
    
    # Process each model
    for model_name in model_names:
        try:
            pred_maps  = load_output_maps(model_name)
            reference_maps_tensor_reshaped = reference_maps.view(reference_maps.shape[0], -1)
        
            #Create residual error map
            residual_maps = pred_maps - reference_maps_tensor_reshaped
            
            # Initialize storage for results by elevation range
            elev_range_results = {
                'rmse_by_time': [],
                'bias_by_time': [],
                'mae_by_time': [],
                'time_hours': np.arange(0, pred_maps.shape[0] * 0.25, 0.25)  # 15 min = 0.25 hours
            }
            
            # For each elevation range, calculate metrics
            for elev_min, elev_max in elev_ranges:
                # Create mask for this elevation range
                if elev_min == float('-inf'):
                    elev_mask = (dem_valid <= elev_max)
                elif elev_max == float('inf'):
                    elev_mask = (dem_valid > elev_min)
                else:
                    elev_mask = (dem_valid > elev_min) & (dem_valid <= elev_max)
                
                # For each timestep, calculate metrics for this elevation range
                rmse_time = []
                bias_time = []
                mae_time = []
                
                for t in range(pred_maps.shape[0]):
                    # Apply elevation mask and only consider wet cells (> 0.3m)
                    # Reshape reference map to match the 2D elevation mask
                    reference_maps_t = reference_maps[t].cpu().numpy()
                    wetcell_mask = reference_maps_t > 0.3
                    combined_mask = elev_mask & wetcell_mask
                    
                    if np.sum(combined_mask) > 0:
                        # Flatten the combined mask to match residual_maps shape
                        flat_mask = combined_mask.flatten()
                        residuals = torch.masked_select(residual_maps[t], torch.tensor(flat_mask, device=residual_maps.device))
                        
                        # Calculate metrics
                        rmse = torch.sqrt(torch.mean(residuals**2)).item()
                        bias = torch.mean(residuals).item()
                        mae = torch.mean(torch.abs(residuals)).item()
                        
                        rmse_time.append(rmse)
                        bias_time.append(bias)
                        mae_time.append(mae)
                    else:
                        # No wet cells in this range at this timestep
                        rmse_time.append(np.nan)
                        bias_time.append(np.nan)
                        mae_time.append(np.nan)
                
                elev_range_results['rmse_by_time'].append(rmse_time)
                elev_range_results['bias_by_time'].append(bias_time)
                elev_range_results['mae_by_time'].append(mae_time)
            
            # Store results for this model
            model_data[model_name] = elev_range_results
            logger.info(f"Processed data for model {model_name}")
            
        except Exception as e:
            logger.error(f"Error processing model {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    if not model_data:
        logger.error("No model data could be processed. Cannot create plots.")
        return
    
    try:
        # Create plot grid: one row per model, one column per elevation range
        n_models = len(model_data)
        n_ranges = len(elev_ranges)
        
        # Create figure and axes
        fig_width = min(24, 4 * n_ranges)
        fig_height = min(15, 3 * n_models)
        fig, axes = plt.subplots(n_models, n_ranges, figsize=(fig_width, fig_height), 
                                sharex='col', sharey='row')
        
        # Ensure axes is always 2D
        if n_models == 1:
            axes = axes.reshape(1, -1)
        if n_ranges == 1:
            axes = axes.reshape(-1, 1)
            
        # Get model display names
        model_display_names = {
            CNN1D_V1: 'Tier-2',
            PICNN1D_V1: 'Tier-3',
            USRR_CNN1D_COMBINED: 'Tier-1',
            HDL_FM_V1: 'Tier-4'
        }
        
        # Find global y limits for consistency
        all_rmse_vals = []
        for model_name, model_results in model_data.items():
            for range_idx in range(len(elev_ranges)):
                rmse_vals = np.array(model_results['rmse_by_time'][range_idx])
                all_rmse_vals.extend(rmse_vals[~np.isnan(rmse_vals)])
        
        # Collect bias values to determine appropriate y-axis range
        all_bias_vals = []
        all_max_vals = []  # To find extreme values
        for model_name, model_results in model_data.items():
            for range_idx in range(len(elev_ranges)):
                bias_vals = np.array(model_results['bias_by_time'][range_idx])
                rmse_vals = np.array(model_results['rmse_by_time'][range_idx])
                all_bias_vals.extend(bias_vals[~np.isnan(bias_vals)])
                
                # Find maximum values for each range to ensure we don't cut off peaks
                if len(rmse_vals[~np.isnan(rmse_vals)]) > 0:
                    all_max_vals.append(np.max(rmse_vals[~np.isnan(rmse_vals)]))
                
        if all_rmse_vals and all_bias_vals:
            # Get absolute maximum RMSE observed in any range
            max_val = max(all_max_vals) if all_max_vals else np.percentile(all_rmse_vals, 99)
            # Use maximum of: 99th percentile * 1.5 or absolute maximum + buffer
            global_y_max = max(np.percentile(all_rmse_vals, 99) * 1.5, max_val * 1.1)
            
            # Find minimum bias value
            min_bias = np.min(all_bias_vals) if all_bias_vals else -0.5
            # Add extra padding for better visualization
            global_y_min = min(-0.5, min_bias * 1.3)
            
            logger.info(f"Plot y-axis limits: {global_y_min:.3f} to {global_y_max:.3f}")
            logger.info(f"Min bias: {min_bias:.3f}, Max RMSE: {max_val:.3f}")
        else:
            global_y_max = 2.0  # Higher upper limit
            global_y_min = -1.0  # More negative to show bias values better
            
        logger.info(f"Plot y-axis limits: {global_y_min:.3f} to {global_y_max:.3f}")
        logger.info(f"Min bias: {min_bias:.3f}, Max RMSE: {max_val:.3f}")
        
        #Plot each model-elevation range combination
        for model_idx, (model_name, model_results) in enumerate(model_data.items()):
            display_name = model_display_names.get(model_name, model_name)
            time_hours = model_results['time_hours']
            
            for range_idx in range(len(elev_ranges)):
                ax = axes[model_idx, range_idx]
                rmse_vals = model_results['rmse_by_time'][range_idx]
                bias_vals = model_results['bias_by_time'][range_idx]
                
                # Plot RMSE line
                # ax.plot(time_hours, rmse_vals, linewidth=2, color='red', label='RMSE')
                
                # Plot bias line
                ax.plot(time_hours, bias_vals, linewidth=2.5, linestyle='-', color='blue', label='Bias')
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
                
                # Set consistent y-axis limits
                ax.set_ylim(global_y_min, global_y_max)
                
                # Increase font size for x and y tick labels
                ax.tick_params(axis='both', which='major', labelsize=16)
                
                # Format x-axis for hours
                import matplotlib.ticker as ticker
                
                # Make x-tick formatting consistent across all subplots
                ax.xaxis.set_major_locator(ticker.MultipleLocator(12))  # Every 12 hours
                
                # Apply the same formatter to all plots to ensure consistency
                def hour_formatter(x, pos):
                    # Format hours (ensuring all ticks at 12-hour intervals are labeled)
                    if x % 12 == 0:
                        return f"{int(x)}h"
                    return ""
                
                # Apply the formatter and make sure ticks are displayed
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(hour_formatter))
                # Force ticks to be visible with proper sizing
                ax.tick_params(axis='x', which='major', length=6, width=1)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Calculate overall metrics
                valid_rmse = [x for x in rmse_vals if not np.isnan(x)]
                valid_bias = [x for x in bias_vals if not np.isnan(x)]
                
                max_bias = np.max(np.abs(valid_bias)) if valid_bias else 0
                mean_bias = np.mean(valid_bias) if valid_bias else 0
                
                # Add metric annotation
                metric_text = f'Max Bias: {max_bias:.2f}m'
                ax.text(0.03, 0.97, metric_text, transform=ax.transAxes, 
                       fontsize=18, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.8))
                
                # Set titles and labels
                if model_idx == 0:  # Top row - add elevation range titles
                    ax.set_title(range_labels[range_idx], fontsize=18, pad=20)  # Increase padding between title and plot
                
                if range_idx == 0:  # Left column - add model labels
                    ax.set_ylabel(f"{display_name}\n Bias(m)", fontsize=18)
                
                if model_idx == n_models - 1:  # Bottom row - add x-labels
                    ax.set_xlabel("Time (hours)", fontsize=18, labelpad=10)  # Add more padding between axis and label
                    # Force x-axis labels to be visible on bottom row
                    plt.setp(ax.get_xticklabels(), visible=True)
                    
                # Event peak vertical line at 34.25 hours
                # peak_hour = 34.25
                # ax.axvline(x=peak_hour, color='black', linestyle='--', alpha=0.5)
                
                # Add legend only to the first plot, positioned lower
                # if model_idx == 0 and range_idx == 0:
                #     ax.legend(loc='lower right', fontsize=18)
        
        # Add overall title
        # fig.suptitle('Model Error by Elevation Percentile (Wet Cells Only)', 
        #             fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.15)
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, 'quality_metrics', "model_error_by_elevation.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model error by elevation plot to {output_path}")
        plt.close(fig)
        
        # Create summary plot - bar chart comparing overall RMSE by elevation range
        plt.figure(figsize=(12, 8))
        
        # Calculate mean RMSE for each elevation range across all models
        mean_rmse_by_range = []
        for range_idx in range(len(elev_ranges)):
            rmse_values = []
            for model_name, model_results in model_data.items():
                rmse_vals = model_results['rmse_by_time'][range_idx]
                # Convert to numpy array first, then apply boolean mask
                rmse_vals_array = np.array(rmse_vals)
                rmse_values.extend(rmse_vals_array[~np.isnan(rmse_vals_array)])
            
            mean_rmse = np.mean(rmse_values) if rmse_values else 0
            mean_rmse_by_range.append(mean_rmse)
        
        # Bar chart for mean RMSE by elevation range
        plt.bar(range_labels, mean_rmse_by_range, color='lightblue', edgecolor='blue', linewidth=1.5)
        
        # Formatting
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Elevation Range', fontsize=14)
        plt.ylabel('Mean RMSE (m)', fontsize=14)
        plt.title('Mean bias over time across six elevation percentile groups', fontsize=18, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save summary plot
        summary_path = os.path.join(OUTPUT_DIR, 'quality_metrics', "mean_rmse_by_elevation_range.png")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved mean RMSE by elevation range plot to {summary_path}")
        
    except Exception as e:
        logger.error(f"Error creating elevation-based analysis plots: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Completed generating elevation-based analysis plots")

def preload_reference_inundation_maps():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    event_id = 1

    if check_inundation_data_cache(event_id):
        inundation_data = torch.load(os.path.join(OUTPUT_DIR, "preprocessed_inundation", f"event_{event_id}_inundation.pt"))
        inundation_data = inundation_data.to(device)
        return inundation_data
    else:
        raise ValueError(f"Inundation data for event {event_id} not found in cache. Please run create_inundation_map_tensors() first.")
    
    
def load_output_maps(model_name):
    ouput_maps_dir = os.path.join(RUN_DIR, 'output_maps', model_name)
    event_inundation_files = glob.glob(f"{ouput_maps_dir}/*.wd")
    event_inundation_files.sort()
    
    # find index which is not available
    for i in range(266):
        expected_file = os.path.join(ouput_maps_dir, "map_{:04d}.wd".format(i))
        if not os.path.exists(expected_file):
            logger.warning(f"Output map file missing: {expected_file}")
            raise FileNotFoundError(f"Output map file missing: {expected_file}")

    ouput_maps = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Process and cache all files for this event
    for i, file in enumerate(event_inundation_files):
        inundation_data = process_inundation_file(file)
        if inundation_data is not None:
            tensor_data = torch.from_numpy(inundation_data).float()
            ouput_maps.append(tensor_data)
    ouput_maps_tensor = torch.stack(ouput_maps).to(device)
    return ouput_maps_tensor
            

def process_inundation_file(file_path):
        try:
            with rio.open(file_path) as src:
                data = src.read(1)
                return data.flatten()
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            return None

def create_confusion_matrix_summary(model_stats, output_dir, idx):
    """Create a summary table of confusion matrix statistics for all models."""
    import pandas as pd
    
    # Convert stats to DataFrame
    stats_data = []
    for model, stats in model_stats.items():
        stats_data.append({
            'Model': model,
            'True Positives': stats['tp'],
            'False Positives': stats['fp'], 
            'False Negatives': stats['fn'],
            'True Negatives': stats['tn'],
            'Hit Rate': f"{stats['hit_rate']:.2f}",
            'CSI': f"{stats['csi']:.2f}",
            'F2 Score': f"{stats['f2_score']:.2f}",
            'F3 Score': f"{stats['f3_score']:.2f}"
        })
    
    df = pd.DataFrame(stats_data)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, f"confusion_matrix_summary{idx}.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved confusion matrix summary to {csv_file}")
    
    # Create visualization of the summary table
    fig, ax = plt.subplots(figsize=(14, len(model_stats) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color model names
    for i in range(1, len(df) + 1):
        table[(i, 0)].set_facecolor('#E3F2FD')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Flood Extent Prediction - Confusion Matrix Summary\n(Threshold: 0.3m)', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    summary_file = os.path.join(output_dir, "confusion_matrix_summary_table.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved confusion matrix summary table to {summary_file}")
    plt.close()
    
    return df


