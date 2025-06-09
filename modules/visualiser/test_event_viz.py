from modules.lib.constants import CARLISLE_DATA_DIR,SIMULATION_DATA_DIR, OUTPUT_DIR, GRAPH_OUTPUT_DIR, RUN_DIR
import os
import logging
import pandas as pd
import glob
from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test Event Visualisation")

def vizualise_test_event():
    # plot_upstream_hydrographs()
    # plot_flood_depth()
    plot_depth_predictions_at_points()
    
def plot_upstream_hydrographs():
    logger.info("Generating upstream hydrograph plots")
    flow_file = os.path.join(CARLISLE_DATA_DIR, "Upstream_Flows_Run1.csv")

    df = pd.read_csv(flow_file)
    df["TimeHours"] = df["Time"] / 3600

    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
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
    output_path = os.path.join(GRAPH_OUTPUT_DIR, "upstream_conditions_testevent.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved combined upstream hydrograph to {output_path}")
    plt.close()

def plot_flood_depth():
    #select the point with the lowest elevation
    # and plot the flood depth at that point
    
    logger.info("Generating flood depth plot at lowest elevation point")
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    simulation_files = glob.glob(os.path.join(SIMULATION_DATA_DIR, "Run1-*.wd"))
    
    # Sort the simulation files by timestep
    simulation_files.sort(key=lambda x: int(os.path.basename(x).split('-')[1].split('.')[0]))
    
    # Read the DEM file
    import numpy as np
    with open(dem_file, 'r') as f:
        # Read header
        ncols = int(f.readline().split()[1])
        nrows = int(f.readline().split()[1])
        xllcorner = float(f.readline().split()[1])
        yllcorner = float(f.readline().split()[1])
        cellsize = float(f.readline().split()[1])
        nodata_value = float(f.readline().split()[1])
        
        # Read elevation data
        elevation_data = []
        for line in f:
            elevation_data.append([float(x) for x in line.split()])
        
        elevation = np.array(elevation_data)
        
        # Create masked array excluding nodata values
        elevation_masked = np.ma.masked_equal(elevation, nodata_value)
        
        # Define the percentiles we want to find (0 = lowest point, 1%, 2%, etc.)
        target_percentiles = [0, 2, 5, 10, 20, 100]  # Lowest point and very low percentiles
        
        # First pass: collect valid elevation values to calculate percentile thresholds
        valid_elevations = []
        # Sample every 10th point to reduce memory usage while still getting representative data
        for row_idx in range(0, elevation_masked.shape[0], 10):
            for col_idx in range(0, elevation_masked.shape[1], 10):
                value = elevation_masked[row_idx, col_idx]
                if value is not np.ma.masked:
                    valid_elevations.append(value)
        
        # Calculate the threshold values for each percentile
        percentile_thresholds = np.percentile(valid_elevations, target_percentiles)
        logger.info(f"Percentile thresholds: {percentile_thresholds}")
        
        # Second pass: find the points closest to each threshold
        points_of_interest = []  # Will store (row, col, elevation) tuples
        closest_points = [(float('inf'), 0, 0) for _ in range(len(target_percentiles))]  # (distance, row, col)
        
        for row_idx in range(elevation_masked.shape[0]):
            for col_idx in range(elevation_masked.shape[1]):
                value = elevation_masked[row_idx, col_idx]
                if value is not np.ma.masked:
                    # Check against each threshold
                    for i, threshold in enumerate(percentile_thresholds):
                        distance = abs(value - threshold)
                        if distance < closest_points[i][0]:
                            closest_points[i] = (distance, row_idx, col_idx)
        
        # Convert to points_of_interest format
        for i, (_, row, col) in enumerate(closest_points):
            elev = elevation_masked[row, col]
            points_of_interest.append((row, col, elev))
            logger.info(f"Selected point for {target_percentiles[i]}th percentile: row={row}, col={col}, elevation={elev:.2f}m")
            
        # Create labels for the points
        point_labels = ["Lowest" if p == 0 else "Highest" if p == 100 else f"{p}th %ile" for p in target_percentiles]
        
        # Save points of interest to CSV
        save_points_to_csv(points_of_interest, point_labels, target_percentiles)
        
        # For backward compatibility with the rest of the code
        lowest_row = points_of_interest[0][0]
        lowest_col = points_of_interest[0][1]
        lowest_value = points_of_interest[0][2]
    
    # Extract the flood depth at all points for each timestep
    timesteps = []
    depths_by_point = [[] for _ in range(len(points_of_interest))]
    
    for sim_file in simulation_files:
        # Extract the timestep from the filename
        timestep = int(os.path.basename(sim_file).split('-')[1].split('.')[0])
        
        # Read the water depth file
        with open(sim_file, 'r') as f:
            # Skip header
            for _ in range(6):
                f.readline()
            
            # Read each row that contains a point of interest
            depth_data_by_row = {}
            for i in range(nrows):
                line = f.readline()
                # If this row contains any of our points of interest, save it
                for p_idx, (row, col, _) in enumerate(points_of_interest):
                    if i == row:
                        if row not in depth_data_by_row:
                            depth_data_by_row[row] = [float(x) for x in line.split()]
            
        # First occurrence of this timestep
        if timestep not in timesteps:
            timesteps.append(timestep)
            
            # Get depth at each point
            for p_idx, (row, col, _) in enumerate(points_of_interest):
                if row in depth_data_by_row:
                    depth = depth_data_by_row[row][col]
                    # Check if it's a nodata value
                    if depth == nodata_value:
                        depth = 0.0
                    depths_by_point[p_idx].append(depth)
                else:
                    # If for some reason we didn't read the row, use 0
                    depths_by_point[p_idx].append(0.0)
    
    # Convert timesteps to hours (each timestep is 15 minutes)
    hours = [timestep * 15 / 60 for timestep in timesteps]
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))  # Taller figure to accommodate layout
    
    # Create a gridspec layout to position plots
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])
    
    # Main axes for the depth plot (takes the entire top row)
    ax_depth = fig.add_subplot(gs[0, :])
    
    # DEM plot - bottom left
    ax_dem = fig.add_subplot(gs[1, 0])
    
    # Main plot: flood depth over time for all points
    colors = ['red', 'orange', 'gold', 'green', 'dodgerblue', 'purple']  # Updated to match the 6 percentiles
    
    for i, depths in enumerate(depths_by_point):
        line, = ax_depth.plot(hours, depths, color=colors[i], linewidth=2.0, 
                label=f"P{i+1}: {point_labels[i]} ({points_of_interest[i][2]:.2f}m)")
        
        # Find and annotate peak depth for points that flood
        if max(depths) > 0.01:
            max_idx = depths.index(max(depths))
            peak_hour = hours[max_idx]
            peak_depth = depths[max_idx]
            
            ax_depth.scatter([peak_hour], [peak_depth], color=colors[i], s=50, zorder=5)
            
            # Position the annotation either above or below based on index
            xytext_offset = (0, 10) if i % 2 == 0 else (0, -25)
            
            ax_depth.annotate(f'P{i+1}: {peak_depth:.2f}m',
                        xy=(peak_hour, peak_depth),
                        xytext=xytext_offset,
                        textcoords='offset points',
                        color=colors[i],
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], alpha=0.7))
    
    # Add horizontal line at zero
    ax_depth.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Format x-axis
    import matplotlib.ticker as ticker
    ax_depth.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax_depth.xaxis.set_minor_locator(ticker.NullLocator())
    
    def hour_formatter(x, pos):
        return f"{int(x)}h" if x == int(x) else ""
    
    ax_depth.xaxis.set_major_formatter(ticker.FuncFormatter(hour_formatter))
    
    # Set title and labels for depth plot
    # ax_depth.set_title('Flood Depth at Different Elevation Percentiles', fontsize=15, fontweight='normal')
    ax_depth.set_xlabel('Time (hours)', fontsize=12)
    
    # Add shaded region to highlight prediction time period
    ax_depth.axvspan(17, 65, alpha=0.5, color="#77D5F4")
    
    # Add label to the shaded region
    y_pos = ax_depth.get_ylim()[1] * 0.9  # Position at 90% of the y-axis height
    ax_depth.text(62, y_pos, 'Prediction Period', 
                 fontsize=15, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    ax_depth.set_ylabel('Water Depth (m)', fontsize=12)
    ax_depth.grid(False)
    
    # Plot the DEM in the bottom left
    masked_dem = np.ma.masked_equal(elevation, nodata_value)
    dem_plot = ax_dem.imshow(masked_dem, cmap='terrain', interpolation='nearest')
    
    # Mark all points of interest on DEM
    for i, (row, col, elev) in enumerate(points_of_interest):
        ax_dem.plot(col, row, 'o', color=colors[i], markersize=6, 
                markeredgecolor='white', markeredgewidth=1)
    
    # Remove ticks from the DEM plot
    ax_dem.set_xticks([])
    ax_dem.set_yticks([])
    ax_dem.set_title("DEM with Sample Points", fontsize=10)
    
    # Add a colorbar for the DEM
    cbar = plt.colorbar(dem_plot, ax=ax_dem, label='Elevation (m)', shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    # Add legend in the bottom right area
    ax_legend = fig.add_subplot(gs[1, 1])
    ax_legend.axis('off')  # Turn off axis for legend area
    
    # Create legend manually in the bottom right position
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=2, marker='o', 
                               label=f"P{i+1}: {point_labels[i]} ({points_of_interest[i][2]:.2f}m)")
                    for i in range(len(points_of_interest))]
    
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(GRAPH_OUTPUT_DIR, "flood_depth_multiple_points.png")
    plt.savefig(output_path, dpi=500, bbox_inches='tight')  # Higher resolution for better quality
    logger.info(f"Saved flood depth plot to {output_path}")
    plt.close(fig)
    return output_path

def save_points_to_csv(points, labels, percentiles):
    """
    Save the points of interest to a CSV file
    
    Args:
        points: List of tuples (row, col, elevation)
        labels: List of labels for each point
        percentiles: List of percentile values for each point
    """
    logger.info("Saving points of interest to CSV")
    
    # Create a DataFrame with the points data
    data = {
        'Point_ID': [f"P{i+1}" for i in range(len(points))],
        'Label': labels,
        'Percentile': percentiles,
        'Row': [p[0] for p in points],
        'Column': [p[1] for p in points],
        'Elevation_m': [p[2] for p in points]
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "points_of_interest.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Points of interest saved to {output_path}")
    
    return output_path

def plot_depth_predictions_at_points():
    """
    Plots the predicted vs. true flood depths at points of interest for each model.
    Reads prediction CSVs generated during model testing and creates comparison plots.
    """
    logger.info("Generating plots comparing predicted vs. true depths at points of interest")
    
    # List of models to include in the comparison
    model_names = ["1DCNN_V1", "PICNN1D_V1"]  # You can add more models to this list
    
    # Create output directory for comparison plots
    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
    
    # Read points of interest file to get metadata about the points
    poi_path = os.path.join(OUTPUT_DIR, "points_of_interest.csv")
    if not os.path.exists(poi_path):
        logger.error(f"Points of interest file not found at {poi_path}")
        return
    
    poi_df = pd.read_csv(poi_path)
    
    # Process each model's predictions
    for model_name in model_names:
        # Look in RUN_DIR instead of OUTPUT_DIR to match where files are saved
        predictions_path = os.path.join(RUN_DIR, model_name, "predictions_at_points_final.csv")
        logger.info(f"Looking for predictions file at {predictions_path}")
        
        if not os.path.exists(predictions_path):
            logger.warning(f"Predictions file not found at {predictions_path}, skipping model {model_name}")
            continue
            
        logger.info(f"Processing predictions for model {model_name}")
        
        # Debug: Print info about the CSV file
        try:
            pred_df = pd.read_csv(predictions_path)
            logger.info(f"Loaded {len(pred_df)} prediction records")
            logger.info(f"Columns in prediction data: {pred_df.columns.tolist()}")
            
            # Print statistics to help diagnose the straight line issue
            for point_id in pred_df['Point_ID'].unique():
                point_data = pred_df[pred_df['Point_ID'] == point_id]
                pred_min = point_data['Predicted_Depth_m'].min()
                pred_max = point_data['Predicted_Depth_m'].max()
                pred_std = point_data['Predicted_Depth_m'].std()
                true_min = point_data['True_Depth_m'].min()
                true_max = point_data['True_Depth_m'].max()
                true_std = point_data['True_Depth_m'].std()
                logger.info(f"Point {point_id} - Predicted: min={pred_min:.3f}, max={pred_max:.3f}, std={pred_std:.3f}")
                logger.info(f"Point {point_id} - True: min={true_min:.3f}, max={true_max:.3f}, std={true_std:.3f}")
                
            # Convert timestep to hours (assuming each timestep is 15 minutes)
            pred_df['TimeHours'] = pred_df['Timestep'] * 15 / 60
            
            # Get unique points
            point_ids = pred_df['Point_ID'].unique()
            
            # Create a figure with subplots - one per point
            n_points = len(point_ids)
            fig, axes = plt.subplots(n_points, 1, figsize=(12, 4*n_points), sharex=True)
            if n_points == 1:
                axes = [axes]  # Make sure axes is a list even with one subplot
                
            # Set a title for the entire figure
            fig.suptitle(f'Model {model_name}: Predicted vs. True Flood Depths', fontsize=16, y=0.99)
            
            # Define colors and markers
            true_color = 'darkblue'
            pred_color = 'crimson'
            
            # Process each point of interest
            for i, point_id in enumerate(point_ids):
                # Filter data for this point
                point_data = pred_df[pred_df['Point_ID'] == point_id]
                
                # Get point metadata from the original points file
                point_meta = poi_df[poi_df['Point_ID'] == point_id]
                if len(point_meta) == 0:
                    logger.warning(f"No metadata found for point {point_id}, using generic label")
                    point_label = f"Point {point_id}"
                    point_elev = 0
                else:
                    point_meta = point_meta.iloc[0]
                    point_label = point_meta['Label']
                    point_elev = point_meta['Elevation_m']
                
                # Sort by time
                point_data = point_data.sort_values('TimeHours')
                
                # Plot true depth
                axes[i].plot(point_data['TimeHours'], point_data['True_Depth_m'], 
                        linewidth=2.0, color=true_color, label='True Depth')
                
                # Plot predicted depth
                axes[i].plot(point_data['TimeHours'], point_data['Predicted_Depth_m'], 
                        linewidth=2.0, color=pred_color, linestyle='--', label='Predicted Depth')
                
                # Find and annotate peak depths if there are variations in the data
                if point_data['True_Depth_m'].std() > 0.01:
                    true_max_idx = point_data['True_Depth_m'].idxmax()
                    true_peak_hour = point_data.loc[true_max_idx, 'TimeHours']
                    true_peak_depth = point_data.loc[true_max_idx, 'True_Depth_m']
                    
                    # Only annotate if there's a significant peak
                    if true_peak_depth > 0.05:
                        axes[i].scatter([true_peak_hour], [true_peak_depth], color=true_color, s=50, zorder=5)
                        axes[i].annotate(f'True peak: {true_peak_depth:.2f}m',
                                    xy=(true_peak_hour, true_peak_depth),
                                    xytext=(10, 10),
                                    textcoords='offset points',
                                    color=true_color,
                                    fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=true_color, alpha=0.7))
                
                if point_data['Predicted_Depth_m'].std() > 0.01:
                    pred_max_idx = point_data['Predicted_Depth_m'].idxmax()
                    pred_peak_hour = point_data.loc[pred_max_idx, 'TimeHours']
                    pred_peak_depth = point_data.loc[pred_max_idx, 'Predicted_Depth_m']
                    
                    # Only annotate if there's a significant peak
                    if pred_peak_depth > 0.05:
                        axes[i].scatter([pred_peak_hour], [pred_peak_depth], color=pred_color, s=50, zorder=5)
                        axes[i].annotate(f'Pred peak: {pred_peak_depth:.2f}m',
                                    xy=(pred_peak_hour, pred_peak_depth),
                                    xytext=(10, -25),
                                    textcoords='offset points',
                                    color=pred_color,
                                    fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=pred_color, alpha=0.7))
                    
                # Calculate error metrics for this point
                rmse = np.sqrt(np.mean((point_data['True_Depth_m'] - point_data['Predicted_Depth_m'])**2))
                mean_error = np.mean(point_data['Error_m'])
                max_error = point_data['Error_m'].abs().max()
                
                # Add error metrics as text
                error_text = f'RMSE: {rmse:.3f}m   Mean Error: {mean_error:.3f}m   Max Error: {max_error:.3f}m'
                axes[i].text(0.02, 0.96, error_text, transform=axes[i].transAxes, 
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Format the subplot
                axes[i].set_title(f"{point_id}: {point_label} (Elevation: {point_elev:.2f}m)")
                axes[i].set_ylabel("Water Depth (m)")
                axes[i].grid(True, alpha=0.3)
                axes[i].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # Set y-axis limits to show more detail if values are small
                if point_data['True_Depth_m'].max() < 0.5 and point_data['Predicted_Depth_m'].max() < 0.5:
                    upper_limit = max(0.5, point_data['True_Depth_m'].max() * 1.2, point_data['Predicted_Depth_m'].max() * 1.2)
                    axes[i].set_ylim(-0.05, upper_limit)
                
                # Add legend
                axes[i].legend(loc='upper right')
            
            # Configure shared x-axis
            axes[-1].set_xlabel("Time (hours)")
            
            # Format x-axis ticks
            import matplotlib.ticker as ticker
            axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(3))
            
            def hour_formatter(x, pos):
                return f"{int(x)}h" if x == int(x) else ""
            
            axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(hour_formatter))
            
            # Adjust layout and save figure
            plt.tight_layout()
            output_path = os.path.join(GRAPH_OUTPUT_DIR, f"{model_name}_depth_predictions_comparison.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction comparison plot to {output_path}")
            plt.close(fig)
            
            # Create a summary figure showing all points on one plot
            plot_summary_comparison(model_name, pred_df, poi_df)
            
        except Exception as e:
            logger.error(f"Error processing predictions for {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    logger.info("Completed generating prediction comparison plots")

def plot_summary_comparison(model_name, pred_df, poi_df):
    """
    Creates a summary plot showing all points in a single graph for easier comparison
    
    Args:
        model_name: Name of the model
        pred_df: DataFrame containing prediction data
        poi_df: DataFrame containing point of interest metadata
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up colors - use different color pairs for each point
        color_pairs = [
            ('red', 'salmon'),
            ('darkgreen', 'lightgreen'),
            ('navy', 'cornflowerblue'),
            ('purple', 'violet'),
            ('brown', 'sandybrown'),
            ('black', 'gray')
        ]
        
        # Get unique points
        point_ids = pred_df['Point_ID'].unique()
        legend_entries = []
        
        for i, point_id in enumerate(point_ids):
            # Get colors for this point
            true_color, pred_color = color_pairs[i % len(color_pairs)]
            
            # Filter data for this point
            point_data = pred_df[pred_df['Point_ID'] == point_id]
            
            # Get point metadata
            point_meta = poi_df[poi_df['Point_ID'] == point_id]
            if len(point_meta) == 0:
                point_label = f"Point {point_id}"
            else:
                point_meta = point_meta.iloc[0]
                point_label = point_meta['Label']
            
            # Sort by time
            point_data = point_data.sort_values('TimeHours')
            
            # Plot true and predicted lines
            true_line, = ax.plot(point_data['TimeHours'], point_data['True_Depth_m'],
                             linewidth=2.0, color=true_color)
            pred_line, = ax.plot(point_data['TimeHours'], point_data['Predicted_Depth_m'],
                             linewidth=1.8, linestyle='--', color=pred_color)
            
            # Add to legend entries
            legend_entries.append((true_line, f"{point_id} True"))
            legend_entries.append((pred_line, f"{point_id} Pred"))
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Water Depth (m)", fontsize=12)
        ax.set_title(f"Model {model_name}: All Points Comparison", fontsize=14)
        
        # Format x-axis
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        
        def hour_formatter(x, pos):
            return f"{int(x)}h" if x == int(x) else ""
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(hour_formatter))
        
        # Add legend
        ax.legend([entry[0] for entry in legend_entries], 
                  [entry[1] for entry in legend_entries], 
                  loc='upper right', ncol=2)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(GRAPH_OUTPUT_DIR, f"{model_name}_all_points_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved summary comparison plot to {output_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating summary comparison plot: {str(e)}")

def visualize_errors():
    """
    Visualizes the spatial distribution of prediction errors across the mesh.
    Creates heatmaps showing error metrics for each cell in the domain.
    """
    logger.info("Generating error visualization for mesh cells")
    
    # List of models to analyze
    model_names = ["1DCNN_V1"]  # Add more models as needed
    
    # Create output directory for error visualizations
    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
    
    # Reference file for raster properties
    ref_file = os.path.join(SIMULATION_DATA_DIR, "Run1-0000.wd")
    if not os.path.exists(ref_file):
        logger.error(f"Reference file not found at {ref_file}")
        return
    
    # Load reference data to get dimensions and metadata
    try:
        import rasterio
        with rasterio.open(ref_file) as src:
            height = src.height
            width = src.width
            transform = src.transform
            nodata_value = src.nodata
            
        # Load DEM for masking and visualization context
        dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
        dem = None
        if os.path.exists(dem_file):
            with rasterio.open(dem_file) as src:
                dem = src.read(1)
                if nodata_value is None:
                    nodata_value = src.nodata
        
        # Process each model
        for model_name in model_names:
            # Load the predictions
            logger.info(f"Processing error visualization for model {model_name}")
            
            # Look for prediction maps
            pred_dir = os.path.join(RUN_DIR, model_name)
            if not os.path.exists(pred_dir):
                logger.warning(f"Prediction directory not found at {pred_dir}")
                continue
                
            # Check if the model saved full prediction maps
            output_maps_dir = os.path.join(pred_dir, "output_maps")
            if not os.path.exists(output_maps_dir):
                logger.warning(f"Output maps directory not found at {output_maps_dir}")
                continue
            
            # Find prediction and ground truth files (using the peak timestep - adjust as needed)
            timestep = 146  # Peak of the flood (can be adjusted based on the event)
            pred_file = os.path.join(output_maps_dir, f"map_{timestep:04d}.wd")
            truth_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{timestep:04d}.wd")
            
            if not os.path.exists(pred_file):
                logger.warning(f"Prediction file not found at {pred_file}")
                continue
                
            if not os.path.exists(truth_file):
                logger.warning(f"Ground truth file not found at {truth_file}")
                continue
            
            # Load prediction and truth data
            with rasterio.open(pred_file) as src:
                pred_data = src.read(1)
                
            with rasterio.open(truth_file) as src:
                truth_data = src.read(1)
            
            # Calculate error metrics
            # 1. Absolute error
            abs_error = np.abs(pred_data - truth_data)
            
            # 2. Relative error (handle division by zero)
            epsilon = 1e-6  # Small value to avoid division by zero
            rel_error = np.zeros_like(abs_error)
            mask = truth_data > 0.01  # Only consider cells with water
            rel_error[mask] = abs_error[mask] / (truth_data[mask] + epsilon)
            rel_error = np.clip(rel_error, 0, 5)  # Limit to reasonable range (0-500%)
            
            # 3. Error direction (over/under prediction)
            error_direction = pred_data - truth_data
            
            # Create visualizations
            # 1. Absolute Error Map
            create_error_map(abs_error, dem, nodata_value, 
                          os.path.join(GRAPH_OUTPUT_DIR, f"{model_name}_absolute_error.png"),
                          "Absolute Error (m)", "plasma", 
                          f"{model_name}: Absolute Error at Peak Flood (t={timestep})")
            
            # 2. Relative Error Map (as percentage)
            create_error_map(rel_error * 100, dem, nodata_value, 
                          os.path.join(GRAPH_OUTPUT_DIR, f"{model_name}_relative_error.png"),
                          "Relative Error (%)", "magma", 
                          f"{model_name}: Relative Error at Peak Flood (t={timestep})")
            
            # 3. Over/Under Prediction Map
            create_error_map(error_direction, dem, nodata_value, 
                          os.path.join(GRAPH_OUTPUT_DIR, f"{model_name}_error_direction.png"),
                          "Error (m)", "coolwarm", 
                          f"{model_name}: Over/Under Prediction at Peak Flood (t={timestep})",
                          center_zero=True)
            
            # 4. Combined visualization with multiple metrics
            create_combined_error_visualization(pred_data, truth_data, abs_error, error_direction, dem, nodata_value,
                                          os.path.join(GRAPH_OUTPUT_DIR, f"{model_name}_combined_error_analysis.png"),
                                          f"{model_name}: Error Analysis at Peak Flood (t={timestep})")
            
            logger.info(f"Completed error visualization for model {model_name}")
            
    except Exception as e:
        logger.error(f"Error in error visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Completed error visualization for mesh cells")

def create_error_map(error_data, dem, nodata_value, output_path, colorbar_label, colormap, title, center_zero=False):
    """
    Creates and saves a visualization of error data across the mesh.
    
    Args:
        error_data: 2D array of error values
        dem: Digital Elevation Model for context (can be None)
        nodata_value: Value to use for masking no-data areas
        output_path: Path to save the output image
        colorbar_label: Label for the colorbar
        colormap: Matplotlib colormap name
        title: Plot title
        center_zero: Whether to center the colormap at zero (for diverging colormaps)
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask nodata values
        masked_error = np.ma.masked_where(error_data == nodata_value, error_data)
        
        # Also mask very small values that are essentially zero
        masked_error = np.ma.masked_where(np.abs(masked_error) < 0.001, masked_error)
        
        # Set colormap normalization
        if center_zero:
            # For diverging colormaps (centered at zero)
            vmax = np.max(np.abs(masked_error))
            norm = plt.Normalize(-vmax, vmax)
        else:
            # For sequential colormaps
            vmin = 0
            vmax = np.percentile(masked_error.compressed(), 99.5)  # 99.5th percentile to avoid outliers
            norm = plt.Normalize(vmin, vmax)
        
        # Plot the error map
        im = ax.imshow(masked_error, cmap=colormap, norm=norm, interpolation='nearest')
        
        # Add hillshade of DEM as background context if available
        if dem is not None:
            from matplotlib.colors import LightSource
            ls = LightSource(azdeg=315, altdeg=45)
            dem_masked = np.ma.masked_equal(dem, nodata_value)
            hillshade = ls.hillshade(dem_masked, vert_exag=1.0)
            ax.imshow(hillshade, cmap='gray', alpha=0.3, interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)
        
        # Set title and remove axes ticks
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add metrics as text
        valid_error = masked_error.compressed()
        if len(valid_error) > 0:
            mean_error = np.mean(valid_error)
            rmse = np.sqrt(np.mean(np.square(valid_error)))
            error_stats = (
                f"Mean Error: {mean_error:.3f}m\n"
                f"RMSE: {rmse:.3f}m\n"
                f"Max Error: {np.max(valid_error):.3f}m"
            )
            ax.text(0.02, 0.02, error_stats, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error map to {output_path}")
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating error map: {str(e)}")

def create_combined_error_visualization(pred_data, truth_data, abs_error, error_direction, dem, nodata_value, output_path, title):
    """
    Creates a comprehensive visualization combining multiple error metrics.
    
    Args:
        pred_data: 2D array of model predictions
        truth_data: 2D array of ground truth values
        abs_error: 2D array of absolute error
        error_direction: 2D array showing over/under prediction
        dem: Digital Elevation Model for context (can be None)
        nodata_value: Value to use for masking no-data areas
        output_path: Path to save the output image
        title: Main title for the figure
    """
    try:
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. True Flood Depth (top left)
        masked_truth = np.ma.masked_where((truth_data == nodata_value) | (truth_data < 0.01), truth_data)
        im1 = axs[0, 0].imshow(masked_truth, cmap='Blues', interpolation='nearest')
        axs[0, 0].set_title("True Flood Depth", fontsize=12)
        cbar1 = plt.colorbar(im1, ax=axs[0, 0], label="Depth (m)", shrink=0.8)
        
        # 2. Predicted Flood Depth (top right)
        masked_pred = np.ma.masked_where((pred_data == nodata_value) | (pred_data < 0.01), pred_data)
        im2 = axs[0, 1].imshow(masked_pred, cmap='Blues', interpolation='nearest')
        axs[0, 1].set_title("Predicted Flood Depth", fontsize=12)
        cbar2 = plt.colorbar(im2, ax=axs[0, 1], label="Depth (m)", shrink=0.8)
        
        # 3. Absolute Error (bottom left)
        masked_abs_error = np.ma.masked_where((abs_error == nodata_value) | (abs_error < 0.01), abs_error)
        im3 = axs[1, 0].imshow(masked_abs_error, cmap='plasma', interpolation='nearest')
        axs[1, 0].set_title("Absolute Error", fontsize=12)
        cbar3 = plt.colorbar(im3, ax=axs[1, 0], label="Error (m)", shrink=0.8)
        
        # 4. Error Direction (bottom right) - Red for over-prediction, Blue for under-prediction
        masked_dir = np.ma.masked_where((error_direction == nodata_value) | (np.abs(error_direction) < 0.01), error_direction)
        vmax = np.max(np.abs(masked_dir))
        im4 = axs[1, 1].imshow(masked_dir, cmap='coolwarm', norm=plt.Normalize(-vmax, vmax), interpolation='nearest')
        axs[1, 1].set_title("Over/Under Prediction", fontsize=12)
        cbar4 = plt.colorbar(im4, ax=axs[1, 1], label="Error (m)", shrink=0.8)
        
        # Remove axis ticks for all subplots
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add global statistics
        valid_mask = (truth_data != nodata_value) & (truth_data > 0.01)
        if np.any(valid_mask):
            # Calculate metrics for wet cells only
            wet_pred = pred_data[valid_mask]
            wet_truth = truth_data[valid_mask]
            
            rmse = np.sqrt(np.mean((wet_pred - wet_truth) ** 2))
            mae = np.mean(np.abs(wet_pred - wet_truth))
            max_error = np.max(np.abs(wet_pred - wet_truth))
            
            # Calculate over/under prediction statistics
            over_pred = np.sum(wet_pred > wet_truth) / len(wet_pred) * 100
            under_pred = np.sum(wet_pred < wet_truth) / len(wet_pred) * 100
            
            stats_text = (
                f"RMSE: {rmse:.3f}m\n"
                f"MAE: {mae:.3f}m\n"
                f"Max Error: {max_error:.3f}m\n"
                f"Over-prediction: {over_pred:.1f}%\n"
                f"Under-prediction: {under_pred:.1f}%"
            )
            
            fig.text(0.5, 0.02, stats_text, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set the main title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined error visualization to {output_path}")
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating combined visualization: {str(e)}")

