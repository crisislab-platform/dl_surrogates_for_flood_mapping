from modules.lib.constants import RUN_DIR, SIMULATION_DATA_DIR, DATA_DIR, OUTPUT_DIR, CNN1D_V1, HDL_FM_V1, PICNN1D_V1, USRR_CNN1D_COMBINED
from modules.datamanager.datamanager import check_inundation_data_cache
import os
import pandas as pd
import logging
import torch
import glob
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name_map = {
    CNN1D_V1: 'Tier-2',
    PICNN1D_V1: 'Tier-3',
    USRR_CNN1D_COMBINED: 'Tier-1',
    HDL_FM_V1: 'Tier-4'
}
    
model_colors = {
        USRR_CNN1D_COMBINED: "#0173B2",  # Blue - safe for all colorblind types
        PICNN1D_V1: "#DE8F05",     # Orange - distinguishable from blue
        "SRR-LSTM": "#CC78BC",    # Light purple/magenta - safe alternative to pink
        CNN1D_V1: "#029E73",       # Green - deuteranopia safe
        HDL_FM_V1: "#D55E00",      # Vermillion/red-orange - protanopia safe
}

# models = [USRR_CNN1D_COMBINED]
models = [USRR_CNN1D_COMBINED,CNN1D_V1,PICNN1D_V1, HDL_FM_V1]

def rmse_stats():
    reference_maps_tensor = preload_reference_inundation_maps()

    # plot_depth_predictions_by_elevation_percentile()
    plot_flood_maps(reference_maps_tensor)
    # rmse_stats_detailed(reference_maps_tensor)
    # plot_sequence_hydrographs_with_windows()
    
def rmse_stats_detailed(reference_maps_tensor):
    
    model_rmses = {}
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
            _, mrmse_per_timestep, mrmse_per_cell = mRMSE(residual_maps, reference_maps_tensor_reshaped)
            
            #calculate NSE per timestep
            nse_per_timestep = NSE(residual_maps, reference_maps_tensor_reshaped)
            
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
                
            model_rmses[model_name] = {
                'rmse_per_timestep': rmse_per_timestep,
                'rmse_per_cell': rmse_per_cell, 
                'mrmse_per_timestep': mrmse_per_timestep,
                'nse_per_timestep': nse_per_timestep,
                'mrmse_per_cell': mrmse_per_cell,
                'hit_rates': hit_rates,
                'csi_scores': csi_scores,
                'f2_scores': f2_scores,
                'f3_scores': f3_scores,
            }
            
            model_name_map = {
                CNN1D_V1: '1DCNN',
                PICNN1D_V1: 'PI1DCNN',
                USRR_CNN1D_COMBINED: 'USRR-1DCNN',
                HDL_FM_V1: 'HDL-FM'
            }
            
            # if model_name == CNN1D_V1 or model_name == USRR_CNN1D_COMBINED:  
            # bootstrap_function(model_name_map.get(model_name), residual_maps, reference_maps_tensor_reshaped, num_samples=10000)
    
    # Plot the RMSE per timestep and RMSE per cell for all models
    # plot_metrics(model_rmses)
    plot_rmse_per_cell(model_rmses)
    
    # # Plot hydrograph and number of wet cells over time
    # plot_hydrograph_and_wet_cells(reference_maps_tensor)
    
    #plot flood extent temporal evolution
    # plot_flood_extent_temporal_evolution(model_rmses)
    
def plot_flood_extent_temporal_evolution(model_rmses):
    """
    Plot temporal evolution of flood extent metrics (Hit Rate, CSI, F2, F3 scores) over time
    """
    # Create a figure for time series plot only
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # === SUBPLOT 1: TIME SERIES PLOT ===
    # Plot Hit Rate per timestep for each model, convert timesteps to hours (15 minutes per timestep)
    timesteps = len(next(iter(model_rmses.values()))['hit_rates'])
    hours = np.arange(0, timesteps * 0.25, 0.25)  # Convert to hours (15 min = 0.25 hours)
    
    # Mark the peak of the event at 34.25 hours
    peak_hour = 34.25
    peak_timestep = int(peak_hour / 0.25)  # Convert hours to timestep index
    if peak_timestep < len(hours):  # Ensure the peak is within the data range
        peak_y_values = [model_data['hit_rates'].cpu().numpy()[peak_timestep] for model_data in model_rmses.values()]
        max_peak_y = max(peak_y_values) if peak_y_values else ax1.get_ylim()[1]  # Use max Hit Rate at peak or current y-limit
        
        # Add vertical line at peak with annotation
        ax1.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.7)
        ax1.annotate('Event Peak', xy=(peak_hour, max_peak_y * 1.05),
                    xytext=(peak_hour + 10, max_peak_y * 1.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                    fontsize=18, fontweight='bold')
        
    # Add peak annotation for each model
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        hit_rates = model_rmses[model_key]['hit_rates'].cpu().numpy()
        ax1.plot(hours, hit_rates, label=model_display_name, color=model_colors[model_key], linewidth=2)
        
        # Annotate peak Hit Rate point for each model
        peak_timestep = np.argmax(hit_rates)
        peak_hour_model = hours[peak_timestep]
        peak_hit_rate = hit_rates[peak_timestep]
        x_offset = 5
        y_offset = 0.05
        
        if model_key == HDL_FM_V1:
            y_offset = -0.02
            x_offset = 3
            
        if model_key == PICNN1D_V1:
            y_offset = -0.02
        
        if model_key == USRR_CNN1D_COMBINED:
            y_offset = -0.02
            x_offset = 3
        # if peak_hit_rate is not None:
            # ax1.annotate(f'{peak_hit_rate:.2f}', xy=(peak_hour_model, peak_hit_rate),
            #              xytext=(peak_hour_model + x_offset, peak_hit_rate + y_offset),
            #              arrowprops=dict(facecolor=model_colors[model_display_name], shrink=0.05, width=1.5, headwidth=7),
            #              fontsize=18, color=model_colors[model_display_name],
            #              bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=model_colors[model_display_name], alpha=0.9))
    
    ax1.set_xlabel('Time (hours)', fontsize=18)
    ax1.set_ylabel('Hit Rate', fontsize=18)
    ax1.set_title('(a) Temporal distribution of Hit rate', fontsize=20, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=18)
    ax1.set_ylim(0, 1.2)  # Hit rate ranges from 0 to 1
    
    # Save the time series figure
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'hit_rate_per_timestep.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Hit Rate per timestep plot at {outfile}")
    
    # Create CSI plot
    plot_csi_temporal(model_rmses)
    
    # Create F2 and F3 scores plot
    plot_f_scores_temporal(model_rmses)

def plot_csi_temporal(model_rmses):
    """
    Plot temporal evolution of Critical Success Index (CSI) over time
    """
    # Create a figure for time series plot only
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # Plot CSI per timestep for each model, convert timesteps to hours (15 minutes per timestep)
    timesteps = len(next(iter(model_rmses.values()))['csi_scores'])
    hours = np.arange(0, timesteps * 0.25, 0.25)  # Convert to hours (15 min = 0.25 hours)
    
    # Mark the peak of the event at 34.25 hours
    peak_hour = 34.25
    peak_timestep = int(peak_hour / 0.25)  # Convert hours to timestep index
    if peak_timestep < len(hours):  # Ensure the peak is within the data range
        peak_y_values = [model_data['csi_scores'].cpu().numpy()[peak_timestep] for model_data in model_rmses.values()]
        max_peak_y = max(peak_y_values) if peak_y_values else ax1.get_ylim()[1]  # Use max CSI at peak or current y-limit
        
        # Add vertical line at peak with annotation
        ax1.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.7)
        ax1.annotate('Event Peak', xy=(peak_hour, max_peak_y * 1.05),
                    xytext=(peak_hour + 10, max_peak_y * 1.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                    fontsize=18, fontweight='bold')
        
    # Add peak annotation for each model
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        csi_scores = model_rmses[model_key]['csi_scores'].cpu().numpy()
        ax1.plot(hours, csi_scores, label=model_display_name, color=model_colors[model_key], linewidth=2)
        
        # Annotate peak CSI point for each model
        peak_timestep = np.argmax(csi_scores)
        peak_hour_model = hours[peak_timestep]
        peak_csi = csi_scores[peak_timestep]
        x_offset = 5
        y_offset = 0.03
        
        if model_key == HDL_FM_V1:
            y_offset = -0.01
            x_offset = 3
            
        if model_key == PICNN1D_V1:
            y_offset = -0.01
        
        if model_key == USRR_CNN1D_COMBINED:
            y_offset = -0.01
            x_offset = 3
        # if peak_csi is not None:
        #     ax1.annotate(f'{peak_csi:.2f}', xy=(peak_hour_model, peak_csi),
        #                  xytext=(peak_hour_model + x_offset, peak_csi + y_offset),
        #                  arrowprops=dict(facecolor=model_colors[model_display_name], shrink=0.05, width=1.5, headwidth=7),
        #                  fontsize=18, color=model_colors[model_display_name],
        #                  bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=model_colors[model_display_name], alpha=0.9))
    
    ax1.set_xlabel('Time (hours)', fontsize=18)
    ax1.set_ylabel('Critical Success Index (CSI)', fontsize=18)
    ax1.set_title('(b) Temproal distribution of CSI', fontsize=20, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=18)
    ax1.set_ylim(0, 1.2)  # CSI ranges from 0 to 1
    
    # Save the time series figure
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'csi_per_timestep.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CSI per timestep plot at {outfile}")

def plot_f_scores_temporal(model_rmses):
    """
    Plot temporal evolution of F2 and F3 scores over time as separate images
    """
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # Plot F2 and F3 scores per timestep for each model, convert timesteps to hours (15 minutes per timestep)
    timesteps = len(next(iter(model_rmses.values()))['f2_scores'])
    hours = np.arange(0, timesteps * 0.25, 0.25)  # Convert to hours (15 min = 0.25 hours)
    
    # Mark the peak of the event at 34.25 hours
    peak_hour = 34.25
    
    # PLOT 1: F2 Scores (separate figure)
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        f2_scores = model_rmses[model_key]['f2_scores'].cpu().numpy()
        ax1.plot(hours, f2_scores, label=model_display_name, color=model_colors[model_key], linewidth=2)
        
    ax1.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Time (hours)', fontsize=18)
    ax1.set_ylabel('$F_{2}$ Score', fontsize=18)
    ax1.set_title('(c) Temporal distribution of $F_{2}$ Score', fontsize=20, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=18)
    ax1.set_ylim(-0.5, 1.2)  # F2 score can be negative
    
    # Save F2 scores figure
    outfile_f2 = os.path.join(OUTPUT_DIR, 'quality_metrics', 'f2_scores_per_timestep.png')
    plt.tight_layout()
    plt.savefig(outfile_f2, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved F2 scores per timestep plot at {outfile_f2}")
    
    # PLOT 2: F3 Scores (separate figure)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        f3_scores = model_rmses[model_key]['f3_scores'].cpu().numpy()
        ax2.plot(hours, f3_scores, label=model_display_name, color=model_colors[model_key], linewidth=2)
        
    ax2.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (hours)', fontsize=18)
    ax2.set_ylabel('$F_{3}$ Score', fontsize=18)
    ax2.set_title('(d) Temporal distribution of $F_{3}$ Score', fontsize=20, pad=15)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=18)
    ax2.set_ylim(-0.5, 1.2)  # F3 score can be negative
    
    # Save F3 scores figure
    outfile_f3 = os.path.join(OUTPUT_DIR, 'quality_metrics', 'f3_scores_per_timestep.png')
    plt.tight_layout()
    plt.savefig(outfile_f3, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved F3 scores per timestep plot at {outfile_f3}")
    
def plot_rmse_per_cell(model_rmses):
    # Plot the spatial distribution of RMSE per cell for each model using DEM
    out_path = os.path.join(OUTPUT_DIR, 'quality_metrics', 'rmse_per_cell_maps')
    os.makedirs(out_path, exist_ok=True)
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    
    # Read DEM for background context
    try:
        with rio.open(dem_file) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            height, width = dem.shape
            dem_nodata = src.nodata  # Get no data value for proper masking
            logger.info(f"DEM dimensions: {height}x{width}")
    except Exception as e:
        logger.error(f"Could not open DEM file {dem_file}: {e}")
        dem = None
        dem_nodata = None
    
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # Plot each model's RMSE spatial distribution
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        logger.info(f"Plotting RMSE map for {model_display_name}")
        
        # Get RMSE per cell and reshape to 2D
        rmse_per_cell = model_rmses[model_key]['rmse_per_cell'].cpu().numpy()
        
        try:
            # Reshape from flattened to 2D grid
            rmse_grid = rmse_per_cell.reshape(611, 951)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot DEM as background if available (match plot_flood_extent_maps)
            if dem is not None:
                # Create a masked version of DEM for better visualization
                dem_masked = np.ma.masked_equal(dem, dem_nodata)  # Consistent mask with plot_flood_extent_maps
                # Using origin='upper' ensures correct orientation (north at top)
                # Don't use extent with the DEM as it can cause scaling issues
                ax.imshow(dem_masked, cmap='gray', alpha=0.3, origin='upper')
            
            # Create colormap for RMSE using a grey-orange-red scheme similar to confusion matrix colors
            from matplotlib.colors import LinearSegmentedColormap
            # Create custom colormap: grey (good) to orange to red (bad)
            colors = [(0.5, 0.5, 0.5, 1.0),    # Grey for low RMSE
                      (1.0, 0.55, 0.0, 1.0),   # Dark orange for medium RMSE
                      (1.0, 0.0, 0.0, 1.0)]    # Red for high RMSE
            cmap = LinearSegmentedColormap.from_list('grey_orange_red', colors)
            
            # Calculate global min and max for consistent scale across all plots
            # Collect all RMSE values for setting consistent scale
            all_rmse_values = []
            for model_key_inner in ordered_models:
                if model_key_inner in model_rmses:
                    all_rmse_values.append(model_rmses[model_key_inner]['rmse_per_cell'].cpu().numpy())
            
            # Combine all values and calculate global max for scaling (exclude zeros)
            all_rmse = np.concatenate([v[v > 0] for v in all_rmse_values])
            vmin = 0
            vmax = np.percentile(all_rmse, 95)  # Use 95th percentile to avoid extreme outliers
            
            # Plot RMSE map with transparency for zeros
            rmse_masked = np.ma.masked_where(rmse_grid == 0, rmse_grid)  # Mask zeros
            # Make sure RMSE plot matches DEM dimensions - use the same dimensions
            im = ax.imshow(rmse_masked, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax, origin='upper')
            
            # Add title and labels
            ax.set_title(f'{model_display_name} Model', fontsize=20)
            
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
            stats_text = f"Mean RMSE: {np.mean(rmse_per_cell[rmse_per_cell > 0]):.3f} m\n"
            stats_text += f"Max RMSE: {np.max(rmse_per_cell):.3f} m"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=18,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
            
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
        fig_cb = plt.figure(figsize=(6, 1))
        ax_cb = fig_cb.add_axes([0.1, 0.2, 0.8, 0.3])  # [left, bottom, width, height]
        
        # Create the same colormap as used in the RMSE maps
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0.5, 0.5, 0.5, 1.0),    # Grey for low RMSE
                 (1.0, 0.55, 0.0, 1.0),   # Dark orange for medium RMSE
                 (1.0, 0.0, 0.0, 1.0)]    # Red for high RMSE
        
        # Create a horizontal colorbar using the same vmin/vmax as the maps
        norm = plt.Normalize(vmin=0, vmax=vmax)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax_cb, orientation='horizontal')
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
    

def plot_rmse_temporal(model_rmses):
    # Create a figure for time series plot only
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # === SUBPLOT 1: TIME SERIES PLOT ===
    # Plot RMSE per timestep for each model, convert timesteps to hours (15 minutes per timestep)
    timesteps = len(next(iter(model_rmses.values()))['rmse_per_timestep'])
    hours = np.arange(0, timesteps * 0.25, 0.25)  # Convert to hours (15 min = 0.25 hours)
    
    # Mark the peak of the event at 36.5 hours
    peak_hour = 34.25
    peak_timestep = int(peak_hour / 0.25)  # Convert hours to timestep index
    if peak_timestep < len(hours):  # Ensure the peak is within the data range
        peak_y_values = [rmse['rmse_per_timestep'].cpu().numpy()[peak_timestep] for rmse in model_rmses.values()]
        max_peak_y = max(peak_y_values) if peak_y_values else ax1.get_ylim()[1]  # Use max RMSE at peak or current y-limit
        
        # Add vertical line at peak with annotation
        ax1.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.7)
        ax1.annotate('Event Peak', xy=(peak_hour, max_peak_y * 1.5),
                    xytext=(peak_hour + 7, max_peak_y * 1.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                    fontsize=18, fontweight='bold')
        
    # Add peak annotation for each model
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        model_display_name = model_name_map.get(model_key, model_key)
        rmse_per_timestep = model_rmses[model_key]['rmse_per_timestep'].cpu().numpy()
        ax1.plot(hours, rmse_per_timestep, label=model_display_name, color=model_colors[model_key], linewidth=2)
        
        # Annotate peak RMSE point for each model
        peak_timestep = np.argmax(rmse_per_timestep)
        peak_hour = hours[peak_timestep]
        peak_rmse = rmse_per_timestep[peak_timestep]
        x_offset = 5
        y_offset = 0.1
        
        if model_key == HDL_FM_V1:
            y_offset = -0.03
            x_offset = 3
            
        if model_key == PICNN1D_V1:
            y_offset = 0.00
            x_offset = 2
        
        if model_key == USRR_CNN1D_COMBINED:
            y_offset = -0.03
            x_offset = 3
            
        if model_key == CNN1D_V1:
            y_offset = -0.02
            x_offset = 3
        if peak_rmse is not None:
            ax1.annotate(f'{peak_rmse:.2f} m', xy=(peak_hour, peak_rmse),
                         xytext=(peak_hour + x_offset, peak_rmse + y_offset),
                         arrowprops=dict(facecolor=model_colors[model_key], shrink=0.05, width=1.5, headwidth=7),
                         fontsize=18, color=model_colors[model_key],
                         bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=model_colors[model_key], alpha=0.9))
    
    ax1.set_xlabel('Time (hours)', fontsize=18)
    ax1.set_ylabel(r'$RMSE_{t}$ (m)', fontsize=18)
    ax1.set_title('c) Temporal distribution of RMSE', fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=18)
    
    # Just keep the time series plot in this function
    # Save the time series figure separately
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'rmse_per_timestep.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE per timestep plot at {outfile}")
    
    # Create a separate boxplot figure
    plot_rmse_boxplot(model_rmses)
    
def plot_box_plot_mrmse_spatial(model_rmses):
    # Plot mRMSE per cell as a box plot including all models in one figure
    plt.figure(figsize=(12, 8))
    
    # Define the specific model order
    ordered_models = models
    
    # Prepare data for box plot in specific order
    box_data = []
    model_names = []
    display_names = []
    
    # Collect data for each model in the specified order
    for model_key in ordered_models:
        if model_key in model_rmses:
            mrmse_per_cell = model_rmses[model_key]['mrmse_per_cell'].cpu().numpy()
            # Filter values for better visualization (exclude zeros and extreme outliers)
            #exclude zeros
            filtered_mrmse_per_cell = mrmse_per_cell[(mrmse_per_cell > 0)]
            
            box_data.append(filtered_mrmse_per_cell)
            model_names.append(model_key)
            display_names.append(model_name_map.get(model_key, model_key))
    
    # Create boxplot with fliers (outliers) shown
    bp = plt.boxplot(box_data, patch_artist=True, notch=True, showfliers=True, 
                    flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markersize': 6, 'linestyle': 'none'})
    
    # Customize boxplot colors
    for i, box in enumerate(bp['boxes']):
        model_key = ordered_models[i]
        box.set(color=model_colors[model_key], linewidth=2)
        box.set(facecolor=model_colors[model_key], alpha=0.7)
    
    # Customize boxplot elements
    for element in ['whiskers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set(color='black', linewidth=2)
    
    # Customize flier (outlier) markers with model-specific colors
    for i, fliers in enumerate(bp['fliers']):
        model_display_name = display_names[i]
        fliers.set(
            marker='o', 
            markerfacecolor=model_colors[model_key],
            markeredgecolor='gray',
            markersize=4,
            alpha=0.5,
            linestyle='none'
        )
            
    stats_csv_file = os.path.join(OUTPUT_DIR, 'quality_metrics', 'mrmse_spatial_stats_summary.csv')
    
    # Get positions of boxes to place text
    positions = range(1, len(display_names) + 1)
    
    # Calculate and add statistics directly to the plot for each box
    for i, data in enumerate(box_data):
        pos = positions[i]
        
        # Calculate statistics
        minimum = np.min(data)
        maximum = np.max(data)
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        mean = np.mean(data)
        
        # IQR on the side of the box
        model_display_name = display_names[i]
        plt.text(pos + 0.3, (q1 + q3)/2, f'IQR: {iqr:.3f}', ha='left', va='center', 
                fontsize=18, rotation=90, color=model_colors[model_key],
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=model_colors[model_key], alpha=0.7))
        
          #save the min, max, median, iqr to rmse_stats_df
        if os.path.exists(stats_csv_file):
            rmse_stats_df = pd.read_csv(stats_csv_file)
        else:
            rmse_stats_df = pd.DataFrame()  
    
        new_row = {
            'Model': model_display_name,
            'Mean_RMSE_m': np.mean([data > 0]),
            'Max_RMSE_m': np.max(data),
            'Min_RMSE_m': np.min(data[data > 0]), 
            'Median_RMSE_m': np.median(data[data > 0]),
            'IQR_RMSE_m': np.percentile(data[data > 0], 75) - np.percentile(data[data > 0], 25)
        }
        
        rmse_stats_df = pd.concat([rmse_stats_df, pd.DataFrame([new_row])], ignore_index=True)
        rmse_stats_df.to_csv(stats_csv_file, index=False)       
    
    plt.xticks(range(1, len(display_names) + 1), display_names, fontsize=18, rotation=0)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel(r'$mRMSE_{s}$ (m)', fontsize=18)
    plt.title(r'b) Distribution of $mRMSE_{s}$', fontsize=20)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # Add some padding to y-axis to make room for the labels
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
    
    # Save the figure
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'mrmse_per_cell_boxplot.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved mRMSE per cell boxplot at {outfile}")
    
def plot_box_rmse_spatial(model_rmses):
     # Plot RMSE per cell as a box plot including all models in one figure
    plt.figure(figsize=(12, 8))
    
    # Define the specific model order
    ordered_models = models
    
    # Prepare data for box plot in specific order
    box_data = []
    model_names = []
    display_names = []
    
    # Collect data for each model in the specified order
    for model_key in ordered_models:
        if model_key in model_rmses:
            rmse_per_cell = model_rmses[model_key]['rmse_per_cell'].cpu().numpy()
            # Filter values for better visualization (exclude zeros and extreme outliers)
            #exclude zeros
            filtered_rmse_per_cell = rmse_per_cell[(rmse_per_cell > 0)]
            
            box_data.append(filtered_rmse_per_cell)
            model_names.append(model_key)
            display_names.append(model_name_map.get(model_key, model_key))
    
    # Create boxplot with fliers (outliers) shown
    bp = plt.boxplot(box_data, patch_artist=True, notch=True, showfliers=True, 
                    flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markersize': 6, 'linestyle': 'none'})
    
    # Customize boxplot colors
    for i, box in enumerate(bp['boxes']):
        model_key = ordered_models[i]
        box.set(color=model_colors[model_key], linewidth=2)
        box.set(facecolor=model_colors[model_key], alpha=0.7)
    
    # Customize boxplot elements
    for element in ['whiskers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set(color='black', linewidth=2)
    
    # Customize flier (outlier) markers with model-specific colors
    for i, fliers in enumerate(bp['fliers']):
        model_display_name = display_names[i]
        fliers.set(
            marker='o', 
            markerfacecolor=model_colors[model_key],
            markeredgecolor='gray',
            markersize=4,
            alpha=0.5,
            linestyle='none'
        )
            
    # Get positions of boxes to place text
    positions = range(1, len(display_names) + 1)
    
    stats_csv_file = os.path.join(OUTPUT_DIR, 'quality_metrics', 'rmse_spatial_stats_summary.csv')
    
    # Calculate and add statistics directly to the plot for each box
    for i, data in enumerate(box_data):
        pos = positions[i]
        
        # Calculate statistics
        minimum = np.min(data)
        maximum = np.max(data)
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        mean = np.mean(data)
        
        # Add the values directly to the plot
        # Maximum value at the top of each whisker
        # plt.text(pos, maximum, f'{maximum:.3f}', ha='center', va='bottom',
        #          fontsize=15, fontweight='bold', color=colors[i % len(colors)])
        
        # # Q3 value (top of box)
        # plt.text(pos + 0.25, q3, f'Q3: {q3:.3f}', ha='left', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # # Median line
        # plt.text(pos - 0.25, median, f'Med: {median:.3f}', ha='right', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # # Mean value (should be shown as a point in the boxplot)
        # plt.text(pos + 0.25, mean, f'μ: {mean:.3f}', ha='left', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # # Q1 value (bottom of box)
        # plt.text(pos - 0.25, q1, f'Q1: {q1:.3f}', ha='right', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # Minimum value at the bottom of each whisker
        # plt.text(pos, minimum, f'{minimum:.3f}', ha='center', va='top',
        #          fontsize=9, fontweight='bold', color=colors[i % len(colors)])
        
        # IQR on the side of the box
        model_display_name = display_names[i]
        plt.text(pos + 0.3, (q1 + q3)/2, f'IQR: {iqr:.3f}', ha='left', va='center', 
                fontsize=18, rotation=90, color=model_colors[model_key],
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=model_colors[model_key], alpha=0.7))
        
        #save the min, max, median, iqr to rmse_stats_df
        if os.path.exists(stats_csv_file):
            rmse_stats_df = pd.read_csv(stats_csv_file)
        else:
            rmse_stats_df = pd.DataFrame()  
    
        new_row = {
            'Model': model_display_name,
            'Mean_RMSE_m': np.mean([data > 0]),
            'Max_RMSE_m': np.max(data),
            'Min_RMSE_m': np.min(data[data > 0]), 
            'Median_RMSE_m': np.median(data[data > 0]),
            'IQR_RMSE_m': np.percentile(data[data > 0], 75) - np.percentile(data[data > 0], 25)
        }
        
        rmse_stats_df = pd.concat([rmse_stats_df, pd.DataFrame([new_row])], ignore_index=True)
        rmse_stats_df.to_csv(stats_csv_file, index=False)       
    
    plt.xticks(range(1, len(display_names) + 1), display_names, fontsize=18 , rotation=0)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel(r'$RMSE_{s}$ (m)', fontsize=18)
    plt.title(r'a) Distribution of $RMSE_{s}$', fontsize=20)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # Add some padding to y-axis to make room for the labels
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
    
    # Save the figure
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'rmse_per_cell_boxplot.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE per cell boxplot at {outfile}")
    
def plot_hydrograph_and_wet_cells(ref_maps_tensor):
    """
    Create a plot with two subplots:
    1. Upstream discharges (hydrograph) for the three inflows
    2. Number of wet cells over time
    
    Args:
        ref_maps_tensor: Tensor of reference inundation data with shape [timesteps, cells]
    """
    logger.info("Creating hydrograph and wet cells plot")
    
    # Load inflow data
    inflow_file = os.path.join(DATA_DIR, 'Upstream_Flows_Run1.csv')
    inflow_data = pd.read_csv(inflow_file)
    inflow_data = inflow_data[8:] # Skip header/metadata rows
    
    upstream1 = inflow_data['Upstream1'].values
    upstream2 = inflow_data['Upstream2'].values
    upstream3 = inflow_data['Upstream3'].values
    
    # Create time axis in hours (15 min intervals = 0.25 hours)
    timesteps = ref_maps_tensor.shape[0]
    hours = np.arange(0, timesteps * 0.25, 0.25)
    
    # Make sure the number of timesteps match
    min_length = min(len(hours), len(upstream1), timesteps)
    hours = hours[:min_length]
    upstream1 = upstream1[:min_length]
    upstream2 = upstream2[:min_length]
    upstream3 = upstream3[:min_length]
    
    # Create figure with two subplots with increased spacing between them
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.2})
    
    # Plot 1: Hydrographs for the three upstream inflows
    ax1.plot(hours, upstream1, color='blue', linewidth=2, label='Upstream 1')
    ax1.plot(hours, upstream2, color='green', linewidth=2, label='Upstream 2')
    ax1.plot(hours, upstream3, color='red', linewidth=2, label='Upstream 3')
    ax1.set_ylabel('Discharge (m³/s)', fontsize=18)
    ax1.set_title('a) Upstream Inflows', fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=18)
    
    # Calculate number of wet cells over time
    # Convert tensor to CPU for numpy operations if needed
    ref_maps_cpu = reference_maps_tensor.cpu().numpy() if isinstance(ref_maps_tensor, torch.Tensor) else ref_maps_tensor
    ref_maps_cpu = ref_maps_cpu.reshape(266, -1)
    
    # Define threshold for wet cells (30cm depth - the same used in mRMSE)
    threshold = 0.3
    
    # Count cells above the threshold at each timestep
    wet_cells = np.sum(ref_maps_cpu[:min_length] > threshold, axis=1)
    
    # Plot 2: Number of wet cells over time
    ax2.plot(hours, wet_cells, color='blue', linewidth=2, label=f'> {threshold}m (wet cells)')
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Number of Wet Cells', fontsize=18)
    ax2.set_title('b) Inundation Extent', fontsize=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=18)
    
    # Find and mark peak discharges for each hydrograph
    peak_hour_upstream1 = hours[np.argmax(upstream1)]
    peak_hour_upstream2 = hours[np.argmax(upstream2)]
    peak_hour_upstream3 = hours[np.argmax(upstream3)]
    peak_discharge1 = np.max(upstream1)
    peak_discharge2 = np.max(upstream2)
    peak_discharge3 = np.max(upstream3)
    
    # Mark peak discharge points with markers
    ax1.plot(peak_hour_upstream1, peak_discharge1, 'o', color='blue', markersize=8, 
             markerfacecolor='white', markeredgewidth=2)
    ax1.plot(peak_hour_upstream2, peak_discharge2, 'o', color='green', markersize=8, 
             markerfacecolor='white', markeredgewidth=2)
    ax1.plot(peak_hour_upstream3, peak_discharge3, 'o', color='red', markersize=8, 
             markerfacecolor='white', markeredgewidth=2)
    
    # Annotate peak discharges
    ax1.annotate(f'{peak_discharge1:.1f}', 
                xy=(peak_hour_upstream1, peak_discharge1),
                xytext=(peak_hour_upstream1 + 4, peak_discharge1 - 100),
                arrowprops=dict(shrink=0.05, width=1),
                fontsize=14, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
                
    # Mark flood event phases for Upstream 1
    # Find indices for different phases
    onset_index = np.where(upstream1 > np.max(upstream1) * 0.1)[0][0]
    rising_limb_index = np.where(upstream1 > np.max(upstream1) * 0.5)[0][0]
    recession_start_index = np.argmax(upstream1) + 20
    recession_end_index = np.where(upstream1 > np.max(upstream1) * 0.1)[0][-1]
    
    # Get corresponding hours
    onset_hour = hours[onset_index]
    rising_limb_hour = hours[rising_limb_index]
    recession_start_hour = hours[recession_start_index]
    recession_end_hour = hours[recession_end_index]
    
    # Add annotations for each phase
    ax1.annotate('Onset', xy=(onset_hour, upstream1[onset_index]),
                xytext=(onset_hour + 5, upstream1[onset_index] + 80),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1),
                fontsize=12, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
    
    ax1.annotate('Rising\nLimb', xy=(rising_limb_hour, upstream1[rising_limb_index]),
                xytext=(rising_limb_hour - 10, upstream1[rising_limb_index] + 50),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1),
                fontsize=12, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
    
    # ax1.annotate('Peak', xy=(peak_hour_upstream1, peak_discharge1),
    #             xytext=(peak_hour_upstream1 - 10, peak_discharge1 + 80),
    #             arrowprops=dict(facecolor='blue', shrink=0.05, width=1),
    #             fontsize=12, color='blue',
    #             bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
    
    ax1.annotate('Recession', xy=(recession_start_hour, upstream1[recession_start_index]),
                xytext=(recession_start_hour + 5, upstream1[recession_start_index] + 50),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1),
                fontsize=12, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
    ax1.annotate(f'{peak_discharge2:.1f}', 
                xy=(peak_hour_upstream2, peak_discharge2),
                xytext=(peak_hour_upstream2 - 8, peak_discharge2 + 20),
                arrowprops=dict(shrink=0.05, width=1),
                fontsize=14, color='green',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='green', alpha=0.8))
    ax1.annotate(f'{peak_discharge3:.1f}', 
                xy=(peak_hour_upstream3, peak_discharge3),
                xytext=(peak_hour_upstream3 + 4, peak_discharge3 + 20),
                arrowprops=dict(shrink=0.05, width=1),
                fontsize=14, color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='red', alpha=0.8))
    
    # Find and mark peak inundated cells
    peak_timestep_wet = np.argmax(wet_cells)
    peak_hour_wet = hours[peak_timestep_wet]
    peak_cells = np.max(wet_cells)
    
    # Log the timestep and hour with maximum inundation
    logger.info(f"Maximum inundation occurs at timestep {peak_timestep_wet} (hour {peak_hour_wet:.2f})")
    logger.info(f"Number of wet cells at peak: {peak_cells:,}")
    
    # Mark peak inundation point with marker
    ax2.plot(peak_hour_wet, peak_cells, 'o', color='blue', markersize=8,
             markerfacecolor='white', markeredgewidth=2)
    
    # Annotate peak inundation with formatted cell count using absolute coordinates
    # Get the y-axis limits to position the text relative to the plot height
    ymin, ymax = ax2.get_ylim()
    y_offset = 0.3 * (ymax - ymin)  # 30% down from the top
    
    ax2.annotate(f'{peak_cells:,}', 
                xy=(peak_hour_wet, peak_cells),  # Point to annotate
                xytext=(peak_hour_wet + 3, ymax - y_offset),  # Text position
                arrowprops=dict(
                    facecolor='blue',
                    shrink=0.05,
                    width=1,
                    headwidth=8
                ),
                fontsize=18, 
                color='blue',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
    
    # Use the time of maximum inundation as the event peak
    peak_hour_event = peak_hour_wet
    
    # Draw event peak time vertical reference line
    ax1.axvline(x=peak_hour_event, color='black', linestyle='--', alpha=0.7)
    ax2.axvline(x=peak_hour_event, color='black', linestyle='--', alpha=0.7)
    
    # Get the y-axis limits for the top plot
    y1_min, y1_max = ax1.get_ylim()
    
    # Position the Event Peak annotation using relative coordinates
    ax1.annotate('Event Peak', 
                xy=(peak_hour_event, y1_max*0.6),
                xytext=(peak_hour_event+5, y1_max*0.50),  # Position text at 75% of the y-axis height
                arrowprops=dict(
                    facecolor='black',
                    shrink=0.05,
                    width=1
                ),
                fontsize=18,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.8))
                
    # Mark the time from peak of upstream 1 to event peak
    if peak_hour_upstream1 < peak_hour_event:
        # Calculate the lag time
        lag_time = peak_hour_event - peak_hour_upstream1
        
        # Draw an arrow connecting the peaks
        ax2.annotate(f'Lag Time: {lag_time:.1f} hours', 
                    xy=(peak_hour_event, ax2.get_ylim()[1]*0.5),
                    xytext=(peak_hour_upstream1 + lag_time/2, ax2.get_ylim()[1]*0.7),
                    arrowprops=dict(
                        arrowstyle='<->',
                        connectionstyle='arc3,rad=.2',
                        color='purple',
                        lw=1.5
                    ),
                    fontsize=10,
                    color='purple',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='purple', alpha=0.8))
        
        # Highlight the lag area with a light fill
        ax2.axvspan(peak_hour_upstream1, peak_hour_event, alpha=0.15, color='purple')
    
    # Add commas to y-axis values for better readability
    from matplotlib.ticker import FuncFormatter
    def format_with_commas(x, pos):
        return f'{int(x):,}'
    
    ax2.yaxis.set_major_formatter(FuncFormatter(format_with_commas))
    
    # Set hour ticks on x-axis (every 6 hours)
    max_hour = max(hours)
    hour_ticks = np.arange(0, max_hour + 6, 6)
    
    # Apply to both axes (ax1 and ax2 since they share x-axis)
    plt.xticks(hour_ticks, [f'{int(h)}' for h in hour_ticks])
    plt.xlabel('Time (hours)', fontsize=18)
    
    # Add minor ticks for every hour on both plots
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
    
    # Ensure output directory exists
    os.makedirs(os.path.join(OUTPUT_DIR, 'quality_metrics'), exist_ok=True)
    
    # Adjust layout and save figure
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'hydrograph_and_wet_cells.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved hydrograph and wet cells plot at {outfile}")

    
def plot_mrmse_temporal(model_rmses):
    # Create a figure for time series plot only
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # === SUBPLOT 1: TIME SERIES PLOT ===
    # Plot masked RMSE per timestep for each model, convert timesteps to hours (15 minutes per timestep)
    timesteps = len(next(iter(model_rmses.values()))['mrmse_per_timestep'])
    hours = np.arange(0, timesteps * 0.25, 0.25)  # Convert to hours (15 min = 0.25 hours)
    
    # Add peak annotation for each model
    for model_key in ordered_models:
        model_display_name = model_name_map.get(model_key, model_key)
        rmse_per_timestep = model_rmses[model_key]['mrmse_per_timestep'].cpu().numpy()
        ax1.plot(hours, rmse_per_timestep, label=model_display_name, color=model_colors[model_key], linewidth=2)
        
        # Annotate peak RMSE point for each model
        #find the timestep with the maximum mRMSE
        peak_timestep = np.argmax(rmse_per_timestep)
        peak_hour = hours[peak_timestep]
        peak_rmse = rmse_per_timestep[peak_timestep]
        if model_key == HDL_FM_V1:
            x_offset = 2
            y_offset = -0.1
        elif model_key == PICNN1D_V1:
            x_offset = 5
            y_offset = 0.2
        elif model_key == USRR_CNN1D_COMBINED:
            x_offset = 5
            y_offset = -0.1
        elif model_key == CNN1D_V1:
            x_offset = 3
            y_offset = -0.1
        if peak_rmse is not None:
            ax1.annotate(f'{peak_rmse:.2f} m', xy=(peak_hour, peak_rmse),
                            xytext=(peak_hour + x_offset, peak_rmse + y_offset),
                            arrowprops=dict(facecolor=model_colors[model_key], shrink=0.05, width=1.5, headwidth=7),
                            fontsize=14, color=model_colors[model_key],
                            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=model_colors[model_key], alpha=0.9))
        
    # Mark the peak of the event at 36.5 hours
    peak_hour = 34.25
    peak_timestep = int(peak_hour / 0.25)  # Convert hours to timestep index
    if peak_timestep < len(hours):  # Ensure the peak is within the data range
        peak_y_values = [rmse['mrmse_per_timestep'].cpu().numpy()[peak_timestep] for rmse in model_rmses.values()]
        max_peak_y = max(peak_y_values) if peak_y_values else ax1.get_ylim()[1]  # Use max RMSE at peak or current y-limit
        
        # Add vertical line at peak with annotation
        ax1.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.7)
        ax1.annotate('Event Peak', xy=(peak_hour, max_peak_y * 2),
                    xytext=(peak_hour + 7, max_peak_y * 2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                    fontsize=18, fontweight='bold')
    
    ax1.set_xlabel('Time (hours)', fontsize=18)
    ax1.set_ylabel(r'$mRMSE_{t}$ (m)', fontsize=18)
    ax1.set_title('d) Temporal distribution of masked RMSE (wet cells only)', fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=18)
    
    # Just keep the time series plot in this function
    # Save the time series figure separately
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'mrmse_per_timestep.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved mRMSE per timestep plot at {outfile}")
    
    # Create a separate boxplot figure
    plot_mrmse_boxplot(model_rmses)


def plot_metrics(model_rmses):
    plot_rmse_temporal(model_rmses)
    plot_box_rmse_spatial(model_rmses)
    plot_mrmse_temporal(model_rmses)
    plot_box_plot_mrmse_spatial(model_rmses)
    plot_sequence_hydrographs_with_windows()

def bootstrap_function(model_name, residual_maps, reference_maps, num_samples=100):
    #overall RMSE
    overall_rmse = torch.sqrt(torch.mean(residual_maps**2))
    logger.info(f"Overall RMSE: {overall_rmse.item()}")
    
    #reshape residual maps back to (timesteps, rows, cols)
    residual_maps = residual_maps.view(residual_maps.shape[0], 611, 951)
    
    # Use GPU for faster processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_maps = residual_maps.to(device).float()
    
    # Optimized block bootstrap
    bootstrap_rmses = np.zeros(num_samples)
    wet_rmse_values = np.zeros(num_samples)
    nse_values = np.zeros(num_samples)
    mae_values = np.zeros(num_samples)
    
    # Define parameters
    temp_block_size = 24
    spatial_block_size = (50, 50)
    
    for i in range(num_samples):
        torch.cuda.empty_cache()
        logger.info(f"Sampling bootstrap iteration {i+1}/{num_samples}")
        
        # 1. Temporal block bootstrap (much faster approach)
        # Generate random indices for temporal blocks all at once
        num_blocks_needed = (266 + temp_block_size - 1) // temp_block_size
        max_start_idx = 266 - temp_block_size + 1
        
        # Generate random start positions for temporal blocks
        block_starts = torch.randint(0, max_start_idx, (num_blocks_needed,), device=device)
        
        # Create index tensor for all positions we need
        indices = []
        for start in block_starts:
            indices.append(torch.arange(start, start + temp_block_size, device=device))
        
        # Concatenate and trim
        time_indices = torch.cat(indices)[:266]
        
        # Use advanced indexing to select all blocks at once (much faster)
        bootstrap_sample = residual_maps[time_indices]
        
        # 2. Spatial block bootstrap using 2D convolution for efficiency
        # We'll use the random sampling of locations but in a batched operation
        bs_height, bs_width = bootstrap_sample.shape[1], bootstrap_sample.shape[2]
        
        # Create an output tensor initialized with zeros on GPU
        spatially_bootstrapped = torch.zeros_like(bootstrap_sample, device=device)
        
        # Use torch.nn.functional.grid_sample for extremely efficient resampling of ALL timesteps at once
        import torch.nn.functional as F
        
        # Convert bootstrap sample to batch format (N,C,H,W) where N=timesteps, C=1
        resampled_data = bootstrap_sample.unsqueeze(1)  # Shape: [timesteps, 1, height, width]
        
        # Create sampling grid coordinates
        # For block bootstrapping, we need to generate a grid that samples blocks rather than individual pixels
        # We'll create a grid that maps from target positions to randomly selected source blocks
        
        # Number of blocks in each dimension
        n_blocks_h = bs_height // spatial_block_size[0]
        n_blocks_w = bs_width // spatial_block_size[1]
        
        # Create random mapping for blocks
        # For each target block position, select a random source block
        block_map_h = torch.randint(0, n_blocks_h, (n_blocks_h,), device=device)
        block_map_w = torch.randint(0, n_blocks_w, (n_blocks_w,), device=device)
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(bs_height, device=device).float() / (bs_height - 1) * 2 - 1,
            torch.arange(bs_width, device=device).float() / (bs_width - 1) * 2 - 1,
            indexing='ij'
        )
        
        # Convert grid coordinates to block indices
        block_idx_y = torch.floor(((grid_y + 1) / 2) * n_blocks_h).long().clamp(0, n_blocks_h-1)
        block_idx_x = torch.floor(((grid_x + 1) / 2) * n_blocks_w).long().clamp(0, n_blocks_w-1)
        
        # Map block indices through our random mapping
        mapped_block_idx_y = block_map_h[block_idx_y]
        mapped_block_idx_x = block_map_w[block_idx_x]
        
        # Convert mapped block indices back to normalized coordinates for grid_sample
        # Add random offsets within the block for smoother borders
        block_size_norm_h = 1.0 / n_blocks_h
        block_size_norm_w = 1.0 / n_blocks_w
        
        # Calculate source coordinates
        src_y = (mapped_block_idx_y.float() + (block_idx_y.float() % 1.0)) * block_size_norm_h * 2 - 1
        src_x = (mapped_block_idx_x.float() + (block_idx_x.float() % 1.0)) * block_size_norm_w * 2 - 1
        
        # Create the sampling grid for grid_sample
        sampling_grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0)  # Shape: [1, H, W, 2]
        
        # Perform the resampling in one efficient operation for all timesteps
        # This is vastly more efficient than the loop-based approach
        spatially_bootstrapped = F.grid_sample(
            resampled_data, 
            sampling_grid.repeat(resampled_data.shape[0], 1, 1, 1),
            mode='nearest',  # Use 'nearest' for block resampling
            align_corners=True
        ).squeeze(1)  # Remove the channel dimension
        
        # No need for count_grid with grid_sample approach as it handles resampling automatically
        
        # Calculate RMSE efficiently with a single operation, with better error handling
        logger.info(f"Calculating RMSE for bootstrap sample {i}")
        logger.info(spatially_bootstrapped.shape)
        
        # Clear CUDA cache to avoid memory issues
        torch.cuda.empty_cache()
        
        try:
            mse = torch.mean(spatially_bootstrapped**2)
            rmse = torch.sqrt(mse)
            
            # Calculate other metrics
            mae = torch.mean(torch.abs(spatially_bootstrapped))
            
            # Get reference data for this bootstrap sample
            bootstrapped_reference = reference_maps.view(reference_maps.shape[0], 611, 951)[time_indices]
            
            # Calculate masked RMSE (only for wet cells)
            wet_rmse, _, _ = mRMSE(spatially_bootstrapped.view(266, -1), 
                             bootstrapped_reference.view(266, -1))
            
            # Calculate NSE
            nse_value = NSE(spatially_bootstrapped.view(266, -1), 
                           bootstrapped_reference.view(266, -1))
            
            # Get standard RMSE value
            rmse_value = rmse.detach().cpu().item()
            
            if rmse_value is None or np.isnan(rmse_value):
                logger.warning(f"Bootstrap RMSE for sample {i} is NaN, skipping this sample.")
                continue
            
            bootstrap_rmses[i] = rmse_value
            wet_rmse_values[i] = wet_rmse
            nse_values[i] = nse_value
            mae_values[i] = mae.detach().cpu().item()
            
            logger.info(f"Bootstrap RMSE for sample {i}: {rmse_value:.4f}, Wet RMSE: {wet_rmse:.4f}, NSE: {nse_value:.4f}")
        except Exception as e:
            logger.warning(f"Error calculating RMSE for bootstrap sample {i}: {e}")
            continue
        
    # Calculate the 95% confidence interval
    bootstrap_rmses_tensor = torch.tensor(bootstrap_rmses)
    lower_ci = np.percentile(bootstrap_rmses, 2.5)
    upper_ci = np.percentile(bootstrap_rmses, 97.5)
    
    logger.info(f"Bootstrap RMSE 95% Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
    logger.info(f"Bootstrap RMSE Mean: {bootstrap_rmses_tensor.mean():.4f}")
    logger.info(f"Bootstrap RMSE Standard Deviation: {bootstrap_rmses_tensor.std():.4f}")
    
    # calculate mean wet RMSE, NSE, MAE and 95% CI
    mean_wet_rmse = np.mean(wet_rmse_values)
    mean_nse = np.mean(nse_values)
    mean_mae = np.mean(mae_values)
    lower_ci_wet_rmse = np.percentile(wet_rmse_values, 2.5)
    upper_ci_wet_rmse = np.percentile(wet_rmse_values, 97.5)
    lower_ci_nse = np.percentile(nse_values, 2.5)
    upper_ci_nse = np.percentile(nse_values, 97.5)
    lower_ci_mae = np.percentile(mae_values, 2.5)
    upper_ci_mae = np.percentile(mae_values, 97.5)
    logger.info(f"Wet RMSE Mean: {mean_wet_rmse:.4f}, 95% CI: [{lower_ci_wet_rmse:.4f}, {upper_ci_wet_rmse:.4f}]")
    logger.info(f"NSE Mean: {mean_nse:.4f}, 95% CI: [{lower_ci_nse:.4f}, {upper_ci_nse:.4f}]")  
    logger.info(f"MAE Mean: {mean_mae:.4f}, 95% CI: [{lower_ci_mae:.4f}, {upper_ci_mae:.4f}]")
    
    
    # Create an enhanced histogram plot
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE curve
    counts, bins, patches = plt.hist(bootstrap_rmses, bins=25, density=True, alpha=0.6, 
                                    color='steelblue', edgecolor='black', label='Bootstrap Samples')
    
    # Add kernel density estimate
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(bootstrap_rmses)
    x = np.linspace(min(bootstrap_rmses), max(bootstrap_rmses), 1000)
    plt.plot(x, kde(x), 'r-', linewidth=2, label='Kernel Density Estimate')
    
    # Add vertical lines for important statistics
    plt.axvline(overall_rmse.item(), color='green', linestyle='dashed', 
                linewidth=2, label=f'Overall RMSE: {overall_rmse.item():.4f}')
    plt.axvline(lower_ci, color='red', linestyle='dotted', 
                linewidth=2, label=f'95% CI Lower: {lower_ci:.4f}')
    plt.axvline(upper_ci, color='red', linestyle='dotted', 
                linewidth=2, label=f'95% CI Upper: {upper_ci:.4f}')
    
    # Add grid and improve aesthetics
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Bootstrap RMSE', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Bootstrap RMSE Distribution for {model_name} (n={num_samples})', fontsize=16)
    plt.legend(fontsize=12)
    
    # Add statistics annotation in the plot
    stats_text = f"Mean: {bootstrap_rmses_tensor.mean():.4f}\n"
    stats_text += f"Std Dev: {bootstrap_rmses_tensor.std():.4f}\n"
    stats_text += f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]"
    
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                fontsize=12, va='top')
    
    # Ensure output directory exists
    os.makedirs(os.path.join(OUTPUT_DIR, 'quality_metrics'), exist_ok=True)
    
    # Save the figure with higher DPI for better quality
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', f'bootstrap_rmse_distribution_{model_name}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved enhanced bootstrap RMSE distribution plot at {outfile}")
    
    out_stat_file = os.path.join(OUTPUT_DIR, 'quality_metrics', f'bootstrap_rmse_stats.csv')
    if os.path.exists(out_stat_file):
        stats_df = pd.read_csv(out_stat_file)
    
    else:
        stats_df = pd.DataFrame(columns=['model', 'mean_rmse', 'std_rmse', 'ci_lower', 'ci_upper',
                                        'overall_rmse', 'mean_wet_rmse', 'ci_lower_wet_rmse', 'ci_upper_wet_rmse',
                                        'mean_nse', 'ci_lower_nse', 'ci_upper_nse',
                                        'mean_mae', 'ci_lower_mae', 'ci_upper_mae',
                                        'num_samples', 'block_size_time', 'block_size_space']) 
    new_row = {
        'model': model_name,
        'mean_rmse': bootstrap_rmses_tensor.mean().item(),
        'std_rmse': bootstrap_rmses_tensor.std().item(),
        'ci_lower': lower_ci,
        'ci_upper': upper_ci, 
        'overall_rmse': overall_rmse.item(),
        'mean_wet_rmse': mean_wet_rmse,
        'ci_lower_wet_rmse': lower_ci_wet_rmse,
        'ci_upper_wet_rmse': upper_ci_wet_rmse,
        'mean_nse': mean_nse,
        'ci_lower_nse': lower_ci_nse,
        'ci_upper_nse': upper_ci_nse,
        'mean_mae': mean_mae,
        'ci_lower_mae': lower_ci_mae,  
        'ci_upper_mae': upper_ci_mae,
        'num_samples': num_samples, 
        'block_size_time': 24,
        'block_size_space': '50x50'
    }
    stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)                
    stats_df.to_csv(out_stat_file, index=False)
    logger.info(f"Saved bootstrap RMSE statistics at {out_stat_file}")

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
    
def mRMSE(residual_maps, reference_maps):
    """
    Calculate masked RMSE - only considers wet cells in reference map
    
    Args:
        residual_maps: Tensor of residuals (pred - ref) with shape [timesteps, cells]
        reference_maps: Tensor of reference values with shape [timesteps, cells]
    """
    # Define threshold for wet cells (30cm depth)
    threshold = 0.3
    
    # Reshape to [timesteps, cells] if needed
    if len(residual_maps.shape) > 2:
        residual_maps = residual_maps.view(residual_maps.shape[0], -1)
    if len(reference_maps.shape) > 2:
        reference_maps = reference_maps.view(reference_maps.shape[0], -1)
    
    # Create binary mask for wet cells in reference (>30cm depth)
    wet_mask = (reference_maps > threshold).float()
    
    # Apply mask to residuals
    masked_residuals = residual_maps * wet_mask
    
    # Only include cells that are wet in the reference
    num_wet_cells = torch.sum(wet_mask, dim=1)
    
    # Calculate RMSE for wet cells only (per timestep)
    wet_mse = torch.sum(masked_residuals**2, dim=1) / torch.clamp(num_wet_cells, min=1)
    wet_rmse_temporal = torch.sqrt(wet_mse)
    
    wet_rmse_spatial = torch.sqrt(torch.mean(masked_residuals**2, dim=0))
    # Return mean over all timesteps
    return torch.mean(wet_rmse_temporal), wet_rmse_temporal, wet_rmse_spatial
    
def NSE(residual_maps, reference_maps):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) from residual maps and reference maps
    
    Args:
        residual_maps: Tensor of residuals (pred - ref) with shape [timesteps, cells]
        reference_maps: Tensor of reference values with shape [timesteps, cells]
    
    Returns:
        NSE value as a scalar
    """
    # Reshape to [timesteps, cells] if needed
    if len(residual_maps.shape) > 2:
        residual_maps = residual_maps.view(residual_maps.shape[0], -1)
    if len(reference_maps.shape) > 2:
        reference_maps = reference_maps.view(reference_maps.shape[0], -1)
        
    # Calculate mean of observed (reference) values
    observed_mean = torch.mean(reference_maps)
    
    # Reconstruct predicted values from residuals and reference
    predicted = reference_maps + residual_maps  # Since residual = predicted - reference
    
    # Calculate NSE components
    numerator = torch.sum((reference_maps - predicted) ** 2)
    denominator = torch.sum((reference_maps - observed_mean) ** 2)
    
    # Handle edge cases
    if denominator.item() == 0.0:
        if numerator.item() == 0.0:
            return 1.0
        else:
            logger.warning("Denominator is zero in NSE calculation, returning NSE as 0")
            return 0.0
            
    # Calculate NSE
    nse = 1.0 - (numerator / denominator)
    return nse.item() 
    
def load_output_maps(model_name):
    ouput_maps_dir = os.path.join(RUN_DIR, 'output_maps', model_name)
    event_inundation_files = glob.glob(f"{ouput_maps_dir}/*.wd")
    event_inundation_files.sort()

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

def plot_rmse_boxplot(model_rmses):
    """
    Create a separate boxplot figure for RMSE distribution
    
    Args:
        model_rmses: Dictionary of model RMSE values
    """
    # Create a new figure for boxplot only
    plt.figure(figsize=(12, 8))
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # Prepare data for box plot in specific order
    box_data = []
    display_names = []
    
    # Collect data for each model in the specified order
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        rmse_per_timestep = model_rmses[model_key]['rmse_per_timestep'].cpu().numpy()
        box_data.append(rmse_per_timestep)
        display_names.append(model_name_map.get(model_key, model_key))
    
    # Create boxplot with fliers (outliers) shown
    bp = plt.boxplot(box_data, patch_artist=True, notch=True, showfliers=True, 
                    flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markersize': 6, 'linestyle': 'none'})
    
    # Customize boxplot colors
    for i, box in enumerate(bp['boxes']):
        model_key = ordered_models[i]
        box.set(color=model_colors[model_key], linewidth=2)
        box.set(facecolor=model_colors[model_key], alpha=0.7)
    
    # Customize boxplot elements
    for element in ['whiskers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set(color='black', linewidth=2)
    
    # Customize flier (outlier) markers with model-specific colors
    for i, fliers in enumerate(bp['fliers']):
        model_display_name = display_names[i]
        fliers.set(
            marker='o', 
            markerfacecolor=model_colors[model_key],
            markeredgecolor='gray',
            markersize=4,
            alpha=0.5,
            linestyle='none'
        )
            
    # Get positions of boxes to place text
    positions = range(1, len(display_names) + 1)
    
    stats_csv_file = os.path.join(OUTPUT_DIR, 'quality_metrics', 'rmse_temporal_stats_box_summary.csv')
    
    # Calculate and add statistics directly to the plot for each box
    for i, data in enumerate(box_data):
        pos = positions[i]
        
        # Calculate statistics
        minimum = np.min(data)
        maximum = np.max(data)
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        mean = np.mean(data)
        
        # Add the values directly to the plot
        # Maximum value at the top of each whisker
        # plt.text(pos, maximum, f'{maximum:.3f}', ha='center', va='bottom',
        #          fontsize=15, fontweight='bold', color=colors[i % len(colors)])
        
        # # Q3 value (top of box)
        # plt.text(pos + 0.25, q3, f'Q3: {q3:.3f}', ha='left', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # # Median line
        # plt.text(pos - 0.25, median, f'Med: {median:.3f}', ha='right', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # # Mean value (should be shown as a point in the boxplot)
        # plt.text(pos + 0.25, mean, f'μ: {mean:.3f}', ha='left', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # # Q1 value (bottom of box)
        # plt.text(pos - 0.25, q1, f'Q1: {q1:.3f}', ha='right', va='center',
        #          fontsize=8, color='black', backgroundcolor='white', alpha=0.8)
        
        # Minimum value at the bottom of each whisker
        # plt.text(pos, minimum, f'{minimum:.3f}', ha='center', va='top',
        #          fontsize=9, fontweight='bold', color=colors[i % len(colors)])
        
        # IQR on the side of the box
        model_display_name = display_names[i]
        plt.text(pos + 0.3, (q1 + q3)/2, f'IQR: {iqr:.3f}', ha='left', va='center', 
                fontsize=18, rotation=90, color=model_colors[model_key],
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=model_colors[model_key], alpha=0.7))
        
        #save the min, max, median, iqr to rmse_stats_df
        if os.path.exists(stats_csv_file):
            rmse_stats_df = pd.read_csv(stats_csv_file)
        else:
            rmse_stats_df = pd.DataFrame()  
    
        new_row = {
            'Model': model_display_name,
            'Mean_RMSE_m': np.mean([data > 0]),
            'Max_RMSE_m': np.max(data),
            'Min_RMSE_m': np.min(data[data > 0]), 
            'Median_RMSE_m': np.median(data[data > 0]),
            'IQR_RMSE_m': np.percentile(data[data > 0], 75) - np.percentile(data[data > 0], 25)
        }
        
        rmse_stats_df = pd.concat([rmse_stats_df, pd.DataFrame([new_row])], ignore_index=True)
        rmse_stats_df.to_csv(stats_csv_file, index=False)       
    
    plt.xticks(range(1, len(display_names) + 1), display_names, fontsize=18 , rotation=0)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel(r'$RMSE_{t}$ (m)', fontsize=18)
    plt.title(r'a) Distribution of $RMSE_{t}$', fontsize=20)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # Add some padding to y-axis to make room for the labels
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
    
    # Ensure output directory exists
    os.makedirs(os.path.join(OUTPUT_DIR, 'quality_metrics'), exist_ok=True)
    
    # Save the figure
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'rmse_temporal_boxplot.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE temporal boxplot at {outfile}")
    
def plot_mrmse_boxplot(model_rmses):
    """
    Create a separate boxplot figure for mRMSE distribution
    
    Args:
        model_rmses: Dictionary of model mRMSE values
    """
    # Create a new figure for boxplot only
    plt.figure(figsize=(12, 8))
    
    # Define specific model order for consistent visualizations
    ordered_models = models
    
    # Prepare data for box plot in specific order
    box_data = []
    display_names = []
    
    # Collect data for each model in the specified order
    for model_key in ordered_models:
        if model_key not in model_rmses:
            continue
            
        mrmse_per_timestep = model_rmses[model_key]['mrmse_per_timestep'].cpu().numpy()
        box_data.append(mrmse_per_timestep)
        display_names.append(model_name_map.get(model_key, model_key))
    
    # Create boxplot with fliers (outliers) shown
    bp = plt.boxplot(box_data, patch_artist=True, notch=True, showfliers=True, 
                    flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markersize': 6, 'linestyle': 'none'})
    
    # Customize boxplot colors
    for i, box in enumerate(bp['boxes']):
        model_key = ordered_models[i]
        box.set(color=model_colors[model_key], linewidth=2)
        box.set(facecolor=model_colors[model_key], alpha=0.7)
    
    # Customize boxplot elements
    for element in ['whiskers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set(color='black', linewidth=2)
    
    # Customize flier (outlier) markers with model-specific colors
    for i, fliers in enumerate(bp['fliers']):
        model_display_name = display_names[i]
        fliers.set(
            marker='o', 
            markerfacecolor=model_colors[model_key],
            markeredgecolor='gray',
            markersize=4,
            alpha=0.5,
            linestyle='none'
        )
            
    stats_csv_file = os.path.join(OUTPUT_DIR, 'quality_metrics', 'mrmse_temporal_stats_box_summary.csv')
    positions = range(1, len(display_names) + 1)
    
    # Calculate and add statistics directly to the plot for each box
    for i, data in enumerate(box_data):
        pos = positions[i]
        
        # Calculate statistics
        minimum = np.min(data)
        maximum = np.max(data)
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        mean = np.mean(data)
        
        # IQR on the side of the box
        model_display_name = display_names[i]
        plt.text(pos + 0.3, (q1 + q3)/2, f'IQR: {iqr:.3f}', ha='left', va='center', 
                fontsize=18, rotation=90, color=model_colors[model_key],
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=model_colors[model_key], alpha=0.7))
        
          #save the min, max, median, iqr to rmse_stats_df
        if os.path.exists(stats_csv_file):
            rmse_stats_df = pd.read_csv(stats_csv_file)
        else:
            rmse_stats_df = pd.DataFrame()  
    
        new_row = {
            'Model': model_display_name,
            'Mean_RMSE_m': np.mean([data > 0]),
            'Max_RMSE_m': np.max(data),
            'Min_RMSE_m': np.min(data[data > 0]), 
            'Median_RMSE_m': np.median(data[data > 0]),
            'IQR_RMSE_m': np.percentile(data[data > 0], 75) - np.percentile(data[data > 0], 25)
        }
        
        rmse_stats_df = pd.concat([rmse_stats_df, pd.DataFrame([new_row])], ignore_index=True)
        rmse_stats_df.to_csv(stats_csv_file, index=False)       
    
    plt.xticks(range(1, len(display_names) + 1), display_names, fontsize=18, rotation=0)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel(r'$mRMSE_{t}$ (m)', fontsize=18)
    plt.title(r'b) Distribution of $mRMSE_{t}$', fontsize=20)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # Add some padding to y-axis to make room for the labels
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
    
    # Ensure output directory exists
    os.makedirs(os.path.join(OUTPUT_DIR, 'quality_metrics'), exist_ok=True)
    
    # Save the figure
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', 'mrmse_temporal_boxplot.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved mRMSE temporal boxplot at {outfile}")


def plot_depth_predictions_by_elevation_percentile():
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
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    try:
        with rio.open(dem_file) as src:
            dem = src.read(1)
            transform = src.transform
            height, width = dem.shape
            dem_nodata = src.nodata
            logger.info(f"Loaded DEM with dimensions {height}x{width}")
    except Exception as e:
        logger.error(f"Could not open DEM file {dem_file}: {e}")
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
                
                if valid_rmse:
                    mean_rmse = np.mean(valid_rmse)
                    mean_bias = np.mean(valid_bias)
                else:
                    mean_rmse = np.nan
                    mean_bias = np.nan
                
                # Add metric annotation
                metric_text = f'Mean Bias: {mean_bias:.3f}m'
                ax.text(0.03, 0.97, metric_text, transform=ax.transAxes, 
                       fontsize=18, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.8))
                
                # Set titles and labels
                if model_idx == 0:  # Top row - add elevation range titles
                    ax.set_title(range_labels[range_idx], fontsize=18, pad=20)  # Increase padding between title and plot
                
                if range_idx == 0:  # Left column - add model labels
                    ax.set_ylabel(f"{display_name}\nError (m)", fontsize=18)
                
                if model_idx == n_models - 1:  # Bottom row - add x-labels
                    ax.set_xlabel("Time (hours)", fontsize=18, labelpad=10)  # Add more padding between axis and label
                    # Force x-axis labels to be visible on bottom row
                    plt.setp(ax.get_xticklabels(), visible=True)
                    
                # Event peak vertical line at 34.25 hours
                # peak_hour = 34.25
                # ax.axvline(x=peak_hour, color='black', linestyle='--', alpha=0.5)
                
                # Add legend only to the first plot, positioned lower
                if model_idx == 0 and range_idx == 0:
                    ax.legend(loc='lower right', fontsize=18)
        
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
        plt.title('Mean RMSE by Elevation Percentile Range', fontsize=16, fontweight='bold')
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

def plot_flood_maps(reference_maps):
    """
    Creates visualizations of flood maps for all available models:
    1. Grid layout with prediction maps for all models.
    2. Grid layout with error maps for all models.
    Each model gets its own row in the grid for easy comparison.
    """
    # List of all models to process
    model_names = models
    peak_timestep = 34.25 * 4
    idx = int(peak_timestep)
    reference_map = reference_maps[idx]
    model_data = {}
    
    # Load DEM for background
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    with rio.open(dem_file) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
        height, width = dem_data.shape
        dem_nodata = src.nodata  # Get no data value for proper masking
        logger.info(f"DEM dimensions: {height}x{width}")
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    

    # Load prediction data for each model
    for model_name in model_names:
        logger.info(f"Loading prediction data for model {model_name}")
        pred_maps_tensor  = load_output_maps(model_name)
        if pred_maps_tensor is None or pred_maps_tensor.shape[0] <= idx:
            logger.warning(f"Skipping model {model_name} due to missing or insufficient data")
            continue
        pred_maps_tensor = pred_maps_tensor[idx].view(-1)
        residual_error = pred_maps_tensor - reference_map.view(-1)
        # Calculate RMSE for this model at the peak timestep
        rmse = torch.sqrt(torch.mean(residual_error**2)).item()
        rmse_wet = torch.sqrt(torch.mean(torch.masked_select(residual_error**2, reference_map.view(-1) > 0.3))).item()
        
        # Create masked versions for better visualization
        pred_np = pred_maps_tensor.cpu().numpy().reshape(dem_data.shape)
        error_np = residual_error.cpu().numpy().reshape(dem_data.shape)
        
        # Mask areas where there's no water in reference or prediction
        reference_np = reference_map.cpu().numpy().reshape(dem_data.shape)
        wet_mask = (reference_np > 0.1) | (pred_np > 0.1)  # Areas where either reference or prediction has water
        masked_error = np.ma.masked_where(~wet_mask, error_np)
        
        model_data[model_name] = {
            'pred': pred_np,
            'error': error_np,
            'masked_error': masked_error,
            'masked_error_wetcells': torch.masked_select(residual_error, reference_map.view(-1) > 0.3).cpu().numpy(),
            'rmse': rmse,
            'rmse_wet': rmse_wet,
        }
    logger.info(f"Creating flood map comparison at timestep {idx} for {len(model_names)} models")
    
    available_models = list(model_data.keys())
    
    # Define output filenames
    output_dir = os.path.join(OUTPUT_DIR, "quality_metrics")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"flood_map_error_comparison_{idx}.png")
    
    # Create custom water colormap: 0 = white, rest = blue gradient
    water_colors = plt.cm.Blues(np.linspace(0.3, 1, 256))  # Start from 20% blue intensity
    for i in range(len(water_colors)):
        water_colors[i, 0:3] = np.clip(water_colors[i, 0:3] * 1.3, 0, 1)
    
    # Set first color (value 0) to pure white
    water_colors[0] = [1.0, 1.0, 1.0, 1.0]  # Pure white for 0 values
    
    dem_cmap = plt.matplotlib.colors.ListedColormap(['lightgray'])
    water_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('enhanced_blues', water_colors)
    
    # Create custom error colormap: 0 = white, negative = blue, positive = red
    from matplotlib.colors import LinearSegmentedColormap
    error_colors = [(0.0, 0.0, 1.0),    # Blue for negative errors (under-prediction)
                    (1.0, 1.0, 1.0),    # White for zero error
                    (1.0, 0.0, 0.0)]    # Red for positive errors (over-prediction)
    error_cmap = LinearSegmentedColormap.from_list('custom_error', error_colors)
    
    all_error_values = []
    for model in available_models:
        all_error_values.extend(model_data[model]['error'])
    if all_error_values:
        p95 = np.percentile(np.abs(all_error_values), 95)
        global_error_max = min(p95 * 1.5, max(abs(np.nanmin(all_error_values)), abs(np.nanmax(all_error_values))))
    else:
        global_error_max = 1.0
    
    try:
        # Calculate figure size - more compact and scientific
        fig_width = 16  # Fixed width for consistency
        fig_height = 12  # Reduced row height for more compact layout
        fig_combined = plt.figure(figsize=(fig_width, fig_height))
        
        # Generate individual model maps
        for model in model_data.keys():
            model_info = model_data[model]
            display_name = model_name_map.get(model, model)
            output_file = os.path.join(output_dir, f"{model}_maps.png")
            
            # Create side-by-side prediction and error maps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left plot: Prediction map with terrain DEM background
            # ax1.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.4, origin='upper')
            im1 = ax1.imshow(model_info['pred'], extent=extent, cmap=water_cmap, alpha=0.8, 
                            vmin=0, vmax=3.0, origin='upper')
            ax1.set_title(f'{display_name} - Prediction', fontsize=14, fontweight='bold')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Add scale bar and north arrow to prediction
            ax1.text(0.95, 0.05, '↑N', transform=ax1.transAxes, fontsize=12, 
                    fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
            
            scalebar_length_m = 500
            scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
            scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
            ax1.plot([scale_x, scale_x + scalebar_length_m], [scale_y, scale_y], 'k-', linewidth=2)
            ax1.text(scale_x + scalebar_length_m/2, scale_y + (extent[3] - extent[2]) * 0.01, 
                    f'{scalebar_length_m}m', ha='center', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            
            # Right plot: Error map (no DEM background - only show errors in wet areas)
            im2 = ax2.imshow(model_info['masked_error'], extent=extent, cmap=error_cmap,
                            vmin=-global_error_max, vmax=global_error_max, alpha=1.0, origin='upper')
            ax2.set_title(f'{display_name} Prediction Error (RMSE: {model_info["rmse"]:.3f}m, masked RMSE: {model_info["rmse_wet"]:.3f}m)', 
                          fontsize=14, fontweight='bold')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # # Calculate and display error statistics
            error_compressed = model_info['masked_error'].flatten()
            if len(error_compressed) > 0:
                over_pred = np.sum(error_compressed > 0) / len(error_compressed) * 100
                under_pred = np.sum(error_compressed < 0) / len(error_compressed) * 100
                
                stats_text = (f"Overestimation: {over_pred:.1f}%\n"
                             f"Underestimation: {under_pred:.1f}%")
                
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=16,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, edgecolor='gray'))
            
            # Calculate and display error statistics
            # error_compressed = model_info['error'].flatten()
            # if len(error_compressed) > 0:
            #     over_pred = np.sum(error_compressed > 0) / len(error_compressed) * 100
            #     under_pred = np.sum(error_compressed < 0) / len(error_compressed) * 100
                
            #     stats_text = (f"Overestimation: {over_pred:.1f}%\n"
            #                  f"Underestimation: {under_pred:.1f}%")
                
            #     ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=16,
            #             verticalalignment='top', horizontalalignment='left',
            #             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, edgecolor='gray'))
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved {display_name} maps to {output_file}")
            plt.close()
        
        # Generate separate colorbar images
        logger.info("Generating separate colorbar images")
        
        # Water depth colorbar
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        
        cax = fig.add_axes([0.1, 0.4, 0.8, 0.2])
        norm = plt.Normalize(0, 3.0)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=water_cmap), 
                         cax=cax, orientation='horizontal')
        cb.set_label('Water Depth (m)', fontsize=14, fontweight='bold')
        cb.ax.tick_params(labelsize=12)
        
        ax.text(0.5, 0.8, 'Water Depth Scale', transform=ax.transAxes, 
               fontsize=16, fontweight='bold', ha='center')
        
        plt.tight_layout()
        depth_cbar_file = os.path.join(output_dir, "depth_colorbar.png")
        plt.savefig(depth_cbar_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved depth colorbar to {depth_cbar_file}")
        plt.close()
        
        # Error colorbar
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        
        cax = fig.add_axes([0.1, 0.4, 0.8, 0.2])
        norm = plt.Normalize(-global_error_max, global_error_max)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=error_cmap), 
                         cax=cax, orientation='horizontal')
        cb.set_label('Error (m)', fontsize=14, fontweight='bold')
        cb.ax.tick_params(labelsize=12)
        
        ax.text(0.5, 0.8, 'Error Scale', transform=ax.transAxes, 
               fontsize=16, fontweight='bold', ha='center')
        # ax.text(0.5, 0.1, 'Blue: Under-prediction (Model < Reference)  |  Red: Over-prediction (Model > Reference)', 
        #        transform=ax.transAxes, fontsize=12, ha='center',
        #        bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', alpha=0.7))
        
        plt.tight_layout()
        error_cbar_file = os.path.join(output_dir, "error_colorbar.png")
        plt.savefig(error_cbar_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved error colorbar to {error_cbar_file}")
        plt.close()
        
        # # Generate reference map
        # logger.info("Generating reference (LISFLOOD) map")
        # fig, ax = plt.subplots(figsize=(10, 8))
        # ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.4, origin='upper')
        # ax.imshow(masked_truth, extent=extent, cmap=water_cmap, alpha=water_alpha, 
        #           vmin=0, vmax=3.0, origin='upper')
        
        # ax.set_title('Reference (LISFLOOD)', fontsize=16, fontweight='bold', pad=15)
        # ax.set_xticks([])
        # ax.set_yticks([])
        
        # # Add scale bar and north arrow
        # ax.text(0.95, 0.05, '↑N', transform=ax.transAxes, fontsize=14, 
        #        fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
        
        # scalebar_length_m = 500
        # scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
        # scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
        # ax.plot([scale_x, scale_x + scalebar_length_m], [scale_y, scale_y], 'k-', linewidth=2)
        # ax.text(scale_x + scalebar_length_m/2, scale_y + (extent[3] - extent[2]) * 0.01, 
        #        f'{scalebar_length_m}m', ha='center', va='bottom', 
        #        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # plt.tight_layout()
        # ref_file = os.path.join(output_dir, f"reference_lisflood_{idx}.png")
        # plt.savefig(ref_file, dpi=300, bbox_inches='tight', facecolor='white')
        # logger.info(f"Saved reference map to {ref_file}")
        # plt.close()
        
        logger.info(f"Generated maps for {len(available_models)} models, 1 reference map, and 2 colorbar images")
        return output_dir
            
    except Exception as e:
        logger.error(f"Error generating individual model maps: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_sequence_hydrographs_with_windows():
    """
    Creates a single plot of all three upstream hydrographs with one highlighted 
    2-hour sequence area from t to t-N, showing a sample input data window used by models.
    """
    logger.info("Generating single sequence hydrograph plot with one highlighted 2-hour input window")
    
    # Load flow data for all three upstream sources
    flow_file = os.path.join(DATA_DIR, "Upstream_Flows_Run1.csv")
    df = pd.read_csv(flow_file)
    df["TimeHours"] = df["Time"] / 3600
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, 'quality_metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define sequence parameters - changed to 2 hours
    sequence_length_hours = 8   # N hours back from current time t
    prediction_start_hour = 0  # When prediction period starts
    prediction_end_hour = 30   # When prediction period ends
    
    # Define upstream sources and colors
    upstream_sources = ['Upstream1', 'Upstream2', 'Upstream3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    source_labels = {
        'Upstream1': 'River Eden (S₁)',
        'Upstream2': 'River Caldew (S₂)', 
        'Upstream3': 'River Petteril (S₃)'
    }
    
    # Create single figure instead of subplots
    fig, ax = plt.subplots(figsize=(14,10))
    
    # Set background colors
    ax.set_facecolor('#f6ffd5ff')  # Light gray background for plot area
    # fig.patch.set_facecolor('#f6ffd5ff')  # White background for figure
    
    # Select one random window time during prediction period
    np.random.seed(42)  # For reproducible results
    window_time = np.random.uniform(prediction_start_hour + sequence_length_hours, 
                                   prediction_end_hour - sequence_length_hours)
    
    # Plot all three hydrographs on the same axes
    for i, source in enumerate(upstream_sources):
        if source not in df.columns:
            continue
            
        # Plot the full hydrograph
        ax.plot(df['TimeHours'], df[source], linewidth=5, color=colors[i], 
                label=source_labels[source])
        
        # Highlight the single sequence window
        t_start = window_time - sequence_length_hours
        t_end = window_time
        
        # Only plot if within data range
        if t_start >= df['TimeHours'].min() and t_end <= df['TimeHours'].max():
            # Get data for this window
            window_mask = (df['TimeHours'] >= t_start) & (df['TimeHours'] <= t_end)
            window_data = df[window_mask]
            
            # Highlight the sequence area with semi-transparent fill
            ax.fill_between(window_data['TimeHours'], 0, window_data[source], 
                          alpha=0.3, color=colors[i], 
                          label=f'{source_labels[source]}: t-{sequence_length_hours}h to t' if i == 0 else "")
            
            # Add vertical lines to mark window boundaries
            ax.axvline(x=t_start, color=colors[i], linestyle='--', alpha=0.7, linewidth=5)
            ax.axvline(x=t_end, color=colors[i], linestyle='-', alpha=0.9, linewidth=5)
        
        # Add S₁, S₂, S₃ annotations near the peak of each hydrograph
        max_value = df[source].max()
        max_time = df.loc[df[source].idxmax(), 'TimeHours']
        
        # Define subscript labels
        subscript_labels = ['S₁', 'S₂', 'S₃']
        
        # Position annotations to avoid overlap
        annotation_positions = [
            (max_time + 10, max_value - 100),    # S₁ - slightly right and up
            (max_time - 5, max_value - 200),    # S₂ - left and up
            (max_time + 10, max_value + 200)     # S₃ - right and down
        ]
        
        x_pos, y_pos = annotation_positions[i]
        ax.annotate(subscript_labels[i], 
                   xy=(max_time, max_value),
                   xytext=(x_pos, y_pos),
                   color=colors[i], 
                   fontsize=50, 
                   fontweight='bold',
                   ha='center',
                   bbox=dict(boxstyle="circle,pad=0.3", fc="white", 
                           edgecolor=colors[i], alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
    # Set y-axis limits for better visualization
    all_max_values = [df[source].max() for source in upstream_sources if source in df.columns]
    if all_max_values:
        global_max = max(all_max_values)
        y_margin = global_max * 0.15
        ax.set_ylim(-y_margin, global_max + y_margin)
    
    # Formatting
    ax.set_ylabel('Flow Rate (m³/s)', fontsize=50)
    
    # Enhanced grid styling
    ax.grid(True, which='major', color='white', linewidth=2.5, alpha=0.8)
    ax.grid(True, which='minor', color='white', linewidth=1, alpha=0.4)
    
    # Make the plot frame/spines more prominent
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#333333')
    
    
    
    # Set custom x-axis ticks with t and t-N labels
    t_start = window_time - sequence_length_hours
    t_end = window_time
    
    # Set only the t and t-N positions as ticks
    ax.set_xticks([t_start, t_end])
    ax.set_yticks([])
    ax.set_xticklabels([f't-N', 't'], fontsize=50)
    
    # Remove x-axis label since we have custom tick labels
    ax.set_xlabel('')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "sequence_hydrographs_single_2h_window.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved single window sequence hydrograph plot to {output_path}")
    plt.close()
    
    return output_path
