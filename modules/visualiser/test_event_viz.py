from modules.lib.constants import CARLISLE_DATA_DIR,SIMULATION_DATA_DIR, OUTPUT_DIR, GRAPH_OUTPUT_DIR
import os
import logging
import pandas as pd
import glob
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test Event Visualisation")

def vizualise_test_event():
    # plot_upstream_hydrographs()
    plot_flood_depth()
    
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

