import os
import pandas as pd
import numpy as np
import glob
import logging
import matplotlib.pyplot as plt
from modules.lib.constants import DATA_DIR, OUTPUT_DIR, PLOTS_OUTPUT_DIR

logger = logging.getLogger("Flow Analysis")

def find_peak_inflow_timestep(run_id=None, output_plot=False):
    
    flow_file_pattern = os.path.join(DATA_DIR, "Upstream_Flows_Run*.csv")
    flow_files = glob.glob(flow_file_pattern)
    
    if not flow_files:
        logger.error(f"No upstream flow files found matching pattern: {flow_file_pattern}")
        return None
    
    # Use the first matching file
    for flow_file in flow_files:
        try:
            # Read the flow data
            df = pd.read_csv(flow_file)
            
            event_id = os.path.basename(flow_file).replace("Upstream_Flows_Run", "").replace(".csv", "")
            logger.info(f"Processing event ID: {event_id}")
            max_flow_1 = df["Upstream1"].max()
            max_flow_2 = df["Upstream2"].max()
            max_flow_3 = df["Upstream3"].max()
            
            # Time of peak inflow
            peak_time_1 = df["Time"][df["Upstream1"].idxmax()]
            peak_time_2 = df["Time"][df["Upstream2"].idxmax()]
            peak_time_3 = df["Time"][df["Upstream3"].idxmax()]
            
            # Convert time from seconds to hours for better readability
            peak_time_1_hours = peak_time_1 / 3600
            peak_time_2_hours = peak_time_2 / 3600
            peak_time_3_hours = peak_time_3 / 3600
            
            # log the results
            logger.info(f"Peak inflow for Upstream1 and event : {max_flow_1} at {peak_time_1_hours} hours")
            logger.info(f"Peak inflow for Upstream2: {max_flow_2} at {peak_time_2_hours} hours")
            logger.info(f"Peak inflow for Upstream3: {max_flow_3} at {peak_time_3_hours} hours")
                 
        except Exception as e:
            logger.error(f"Error finding peak inflow timestep: {e}")
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
    ref_maps_cpu = ref_maps_tensor.cpu().numpy() if isinstance(ref_maps_tensor, torch.Tensor) else ref_maps_tensor
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
    ax1.set_xlabel('Time (hours)', fontsize=18)
    
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
            ax1.text(0.95, 0.05, '↑N', transform=ax1.transAxes, fontsize=14, 
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
        #        bbox=dict(boxstyle="round,pad=0.3", fc='lightgray', alpha=0.7))
        
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
    

def plot_upstream_hydrographs():
    logger.info("Generating upstream hydrograph plots")
    flow_file = os.path.join(DATA_DIR, "Upstream_Flows_Run1.csv")

    df = pd.read_csv(flow_file)
    df["TimeHours"] = df["Time"] / 3600

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    upstream_sources = ['Upstream1', 'Upstream2', 'Upstream3']

    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    legend_entries = []
    for i, source in enumerate(upstream_sources):
        if source in df.columns:
            time_col = 'TimeHours' if 'TimeHours' in df.columns else 'Time'
            
            display_name = ""
            if source == 'Upstream1':
                display_name = 'River Eden (S₁)'
            elif source == 'Upstream2':
                display_name = 'River Caldew (S₂)'
            elif source == 'Upstream3':
                display_name = 'River Petteril (S₃)'
                
            line, = plt.plot(df[time_col], df[source], linewidth=2.5, color=colors[i], 
                        label=display_name)
            
            legend_entries.append(line)
            
            max_value = df[source].max()
            max_time = df.loc[df[source].idxmax(), time_col]
            plt.scatter([max_time], [max_value], color=colors[i], s=60)
            
            # Customize annotation position based on the source
            if source == 'Upstream1':  # River Eden - move to left
                xytext_pos = (-150,6)  # Negative x offset to place on left
                horizontalalignment = 'right'
            else:
                xytext_pos = (10, 10 + i*20)  # Original right-side positioning
                horizontalalignment = 'left'

            plt.annotate(f'{display_name}: {max_value:.1f} m³/s',
                        xy=(max_time, max_value),
                        xytext=xytext_pos,
                        textcoords='offset points',
                        color=colors[i],
                        fontsize=11,
                        horizontalalignment=horizontalalignment,
                        arrowprops=dict(arrowstyle='->', color=colors[i]),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], alpha=0.8))

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Flow Rate (m³/s)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.legend(handles=legend_entries, title="", loc='upper right', 
              prop={'size': 12}, title_fontsize=14)
    output_path = os.path.join(PLOTS_OUTPUT_DIR, "upstream_conditions_testevent.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved combined upstream hydrograph to {output_path}")
    plt.close()