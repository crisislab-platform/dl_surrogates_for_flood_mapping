import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR as DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR, RUN_DIR, GRAPH_OUTPUT_DIR
import logging
import rasterio
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import glob
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
from pathlib import Path
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import gdal_asarray


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Visualiser")

def plot(plot_type, file, run_id):
    if plot_type == 'upstream':
        plot_upstream_flows(file)
    elif plot_type == 'rep_locations':
        visualise_rep_locations(file, run_id)
    else:
        logger.error(f"Invalid plot type: {plot_type}")
        exit(1)
        
def plot_upstream_flows(csv_file, show_plots=True):
    
    """
    Plot beautiful visualizations of upstream flow data from a CSV file.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing upstream flow data
    output_dir : str, optional
        Directory to save the plots. If None, plots won't be saved.
    show_plots : bool, optional
        Whether to display the plots. Default is True.
    
    Returns:
    --------
    dict
        Dictionary containing the figure objects
    """
    # Load the data
    file_path = os.path.join(DATA_DIR, csv_file)
    output_dir = OUTPUT_DIR
    df = pd.read_csv(file_path)
    
    # Convert time from seconds to hours
    df['TimeHours'] = df['Time'] / 3600
    
    # Setup aesthetics
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Nice blue, orange, green colors
    
    # Create a dictionary to store figure objects
    figures = {}
    
    # Create separate plots for each upstream column
    for idx, column in enumerate(['Upstream1', 'Upstream2', 'Upstream3']):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data with a beautiful style
        ax.plot(df['TimeHours'], df[column], linewidth=2.5, color=colors[idx])
        
        # Fill area under the curve with transparency
        ax.fill_between(df['TimeHours'], df[column], alpha=0.3, color=colors[idx])
        
        # Set titles and labels
        ax.set_title(f'Hydrograph', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel(f'Flow Rate (m$^3$/s)', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Improve aesthetics
        plt.tight_layout()
        
        # Add annotations for max and min values
        max_value = df[column].max()
        max_time = df.loc[df[column].idxmax(), 'TimeHours']
        min_value = df[column].min()
        min_time = df.loc[df[column].idxmin(), 'TimeHours']
        
        # ax.annotate(f'Max: {max_value:.2f}', 
        #            xy=(max_time, max_value),
        #            xytext=(10, 15),
        #            textcoords='offset points',
        #            arrowprops=dict(arrowstyle='->', lw=1.5),
        #            fontsize=10)
                   
        # Store figure
        figures[column] = fig
        
        # Save the figure if output_dir is specified
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig.savefig(os.path.join(output_dir, f'{column}_flow.png'), dpi=300, bbox_inches='tight')
    
    # Show the plots if specified
    if show_plots:
        plt.show()
    
    return figures

def visualise_rep_locations(shp_file, run_id, sampling_dist=None):
    if shp_file is None:
        logger.error("Invalid shapefile path provided!")
        return

    file_path = f"{RUN_DIR}/{run_id}/{shp_file}"
    output_dir = OUTPUT_DIR
    dem_asc_file = f"{DATA_DIR}/Carlisle_5m.asc"
    
    # Use geopandas to read the shapefile instead of rasterio/fiona
    try:
        gdf = gpd.read_file(file_path)
        logger.info(f"Loaded shapefile with {len(gdf)} representative locations")
        logger.info(f"Columns: {gdf.columns}")
        logger.info(f"CRS: {gdf.crs}")
        logger.info(f"Bounds: {gdf.total_bounds}")
    except Exception as e:
        logger.error(f"Error reading shapefile: {e}")
        return None
    
    # Extract point coordinates and ids from the GeoDataFrame
    points = list(gdf.geometry.apply(lambda geom: (geom.x, geom.y)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15,10))
    
    # Plot the representative locations
    #set bounds for the plot
    #x origin 
    
    with rasterio.open(dem_asc_file) as src:
        # Get the DEM data
        dem_data = src.read(1)
        
        # Get the spatial extent for proper alignment
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        logger.info(f"DEM bounds: {extent}")
        
        im = ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.5, origin='upper')
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Elevation (m)')
        
        # Add grid lines based on sampling distance
        if sampling_dist:
            pixel_width = src.transform[0]  # Assuming square pixels
            grid_spacing_meters = sampling_dist
            grid_spacing = grid_spacing_meters / pixel_width
            
            logger.info(f"Pixel width: {pixel_width}")
            logger.info(f"Grid spacing (meters): {grid_spacing_meters}")
            logger.info(f"Grid spacing (pixels): {grid_spacing}")
            
            x_start = src.bounds.left
            x_end = src.bounds.right
            y_start = src.bounds.bottom
            y_end = src.bounds.top
            
            num_cols = int((x_end - x_start) / grid_spacing_meters)
            num_rows = int((y_end - y_start) / grid_spacing_meters)
            
            logger.info(f"Number of expected columns: {num_cols}")
            logger.info(f"Number of expected rows: {num_rows}")
            
            # Adjust the starting point for y_ticks to draw from the top
            y_ticks = np.arange(y_end, y_start, -grid_spacing_meters)
            
            x_ticks = np.arange(x_start, x_end, grid_spacing_meters)
            
            # Create two different grid systems: sampling_dist grid and 5m grid
            
            # First add the 5m grid with lower alpha
            ax.grid(True, color='gray', alpha=0.2, linestyle='-', linewidth=0.3)
            
            # Then set the sampling_dist grid ticks and draw that grid
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.grid(True, color='black', alpha=0.5, linestyle='-', linewidth=0.5)

    x_coords, y_coords = zip(*points) if points else ([], [])
    logger.info(f"X coords len {len(x_coords)}")
    ax.scatter(x_coords, y_coords, color='red', s=40,  # Reduced point size
               edgecolor='black', linewidth=1, alpha=0.7)

    # Set title with sampling distance included
    ax.set_title(f'Representative Locations (sampling distance: {shp_file.split("_")[1].split(".")[0]})', fontsize=14)
    # plt.tight_layout()
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f'representative_locations.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {output_path}")
    
    plt.show()
    return fig

def plot_training_history(history, run_dir):
    """Save model training history plots"""
    if not history:
        logger.error("Model history not found")
        return None

    logger.info("Plotting model history")
    if isinstance(history, dict):
        # PyTorch style history dict
        plt.figure()
        if 'loss' in history and 'val_loss' in history:
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(os.path.join(run_dir, 'model_history.png'))
            plt.close()
    else:
        logger.warning(f"Unknown history format: {type(history)}")
        
        
def plot_upstream_conditions(event_id):
    """
    Plot the upstream conditions for a given event.
    
    Args:
        event_id (str): The ID of the event to plot
    """
    logger.info(f"Plotting upstream conditions for event {event_id}")
    
    # Define paths to data (adjust as needed)
    
    upstream_conditions = os.path.join(DATA_DIR, f"Upstream_Flows_Run{event_id}.csv")
    try:
        # Load data
        df = pd.read_csv(upstream_conditions)
        
        # Plot all three upstream flows on the same figure
        plt.figure(figsize=(12, 6))
        
        # Use Time column if it exists, otherwise use TimeHours if that exists
        time_col = 'Time' if 'Time' in df.columns else 'time' if 'time' in df.columns else None
        
        if time_col:
            # Convert time to hours if it's in seconds and very large
            if df[time_col].max() > 10000:  # Likely in seconds
                time_values = df[time_col] / 3600
                x_label = 'Time (hours)'
            else:
                time_values = df[time_col]
                x_label = 'Time'
        else:
            # If no time column exists, use index
            time_values = df.index
            x_label = 'Time Step'
        
        # Plot each upstream flow with a different color
        if 'Upstream1' in df.columns:
            plt.plot(time_values, df['Upstream1'], 'b-', linewidth=2, label='Upstream 1')
        if 'Upstream2' in df.columns:
            plt.plot(time_values, df['Upstream2'], 'r-', linewidth=2, label='Upstream 2')
        if 'Upstream3' in df.columns:
            plt.plot(time_values, df['Upstream3'], 'g-', linewidth=2, label='Upstream 3')
            
        plt.xlabel(x_label)
        plt.ylabel('Flow (m³/s)')
        plt.title(f'Upstream Flow Conditions')
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_dir = Path("output/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR,f"upstream_flow_event_{event_id}.png")
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Upstream conditions plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting upstream conditions: {e}")
        raise


def plot_boundary_information():
    """
    Visualizes the boundary conditions from the BCI file on the DEM.
    Shows the different types of boundaries (water level boundary, upstream1, upstream2, upstream3).
    """
    dem_file = f"{DATA_DIR}/Carlisle_5m.asc"
    bci_file = f"{DATA_DIR}/carlisle.bci"
    
    output_file = os.path.join(OUTPUT_DIR, "boundary_conditions.png")
    
    logger.info(f"Visualizing boundary conditions from {bci_file}")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Load and display DEM as background
    try:
        with rasterio.open(dem_file) as src:
            # Get the DEM data
            dem_data = src.read(1)
            
            # Get the spatial extent for proper alignment
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            
            logger.info(f"DEM bounds: {extent}")
            
            # Display DEM with terrain colormap
            im = ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.7, origin='upper')
            cbar = fig.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Elevation (m)')
    except Exception as e:
        logger.error(f"Error loading DEM file: {e}")
        return
    
    # Parse BCI file and extract boundary conditions
    upstream1_points = []
    upstream2_points = []
    upstream3_points = []
    
    try:
        with open(bci_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'): 
                    continue
                
                parts = line.split()
        
                if parts[0] == 'P':  # Point source
                    # P x y QVAR upstream#
                    x, y = float(parts[1]), float(parts[2])
                    if parts[4] == 'upstream1':
                        upstream1_points.append((x, y))
                    elif parts[4] == 'upstream2':
                        upstream2_points.append((x, y))
                    elif parts[4] == 'upstream3':
                        upstream3_points.append((x, y))
    
        logger.info(f"Found {len(upstream1_points)} upstream1 points")
        logger.info(f"Found {len(upstream2_points)} upstream2 points")
        logger.info(f"Found {len(upstream3_points)} upstream3 points")
    
    except Exception as e:
        logger.error(f"Error parsing BCI file: {e}")
        return
    
    # Plot upstream1 points
    if upstream1_points:
        x, y = zip(*upstream1_points)
        ax.scatter(x, y, color='blue', s=100, edgecolor='black', linewidth=1, 
                 label='Upstream 1', marker='o', alpha=0.8)
        
        # Annotate one of the upstream1 points for reference
        x_ref, y_ref = upstream1_points[0]
        ax.annotate(f"Upstream 1\n({x_ref}, {y_ref})",
                   xy=(x_ref, y_ref),
                   xytext=(20, 20),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", color="black"))
    
    # Plot upstream2 points
    if upstream2_points:
        x, y = zip(*upstream2_points)
        ax.scatter(x, y, color='red', s=100, edgecolor='black', linewidth=1, 
                 label='Upstream 2', marker='s', alpha=0.8)  # Square marker
        
        # Annotate one of the upstream2 points
        x_ref, y_ref = upstream2_points[0]
        ax.annotate(f"Upstream 2\n({x_ref}, {y_ref})",
                   xy=(x_ref, y_ref),
                   xytext=(20, -40),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", color="black"))
    
    # Plot upstream3 points
    if upstream3_points:
        x, y = zip(*upstream3_points)
        ax.scatter(x, y, color='green', s=100, edgecolor='black', linewidth=1, 
                 label='Upstream 3', marker='^', alpha=0.8)  # Triangle marker
        
        # Annotate one of the upstream3 points
        x_ref, y_ref = upstream3_points[0]
        ax.annotate(f"Upstream 3\n({x_ref}, {y_ref})",
                   xy=(x_ref, y_ref),
                   xytext=(-80, -40),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", color="black"))
    
    # Add title and legend
    ax.set_title('Model Boundary Conditions', fontsize=16, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.legend(loc='upper right')
    
    # Save figure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Boundary conditions visualization saved to {output_file}")
    
    plt.close()
    
def create_flood_animation():
    output_filename="flood_simulation.gif"
    interval=200 
    dpi=100
    file_path = os.path.join(SIMULATION_DATA_DIR, "Run1-*.wd")
    wd_files = sorted(glob.glob(file_path))
    
    logger.info(f"Found {len(wd_files)} .wd files")
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(OUTPUT_DIR, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # DEM file for background (modify path as needed)
    dem_file = os.path.join(CARLISLE_DATA_DIR, "Carlisle_5m.asc")
    
    # Load DEM for background
    try:
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            # Get spatial extent for proper alignment
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            transform = src.transform
            dem_nodata = src.nodata
    except Exception as e:
        logger.error(f"Error loading DEM file: {e}")
        return None
    
    # Create frames
    frame_files = []
    for i, wd_file in enumerate(wd_files):
        logger.info(f"Processing file {i+1}/{len(wd_files)}: {os.path.basename(wd_file)}")
        
        try:
            # Extract time from filename (assuming format includes time information)
            time_str = Path(wd_file).stem.split('_')[-1]  # Adjust based on actual naming convention
            
            # Read water depth file
            with rasterio.open(wd_file) as src:
                wd_data = src.read(1)
                wd_nodata = src.nodata
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot DEM as background with terrain colormap
            ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.7, origin='upper')
            
            # Create a masked array for water depths to handle nodata values
            masked_wd = np.ma.masked_where((wd_data == wd_nodata) | (wd_data <= 0.01), wd_data)
            
            # Plot water depth with blue colormap
            im = ax.imshow(masked_wd, extent=extent, cmap='Blues', alpha=0.8, 
                          vmin=0, vmax=np.max(masked_wd) if not np.ma.is_masked(np.max(masked_wd)) else 3, 
                          origin='upper')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Water Depth (m)')
            
            # Add title with time information
            ax.set_title(f'Flood Simulation - Time: {time_str}', fontsize=14)
            
            # Save frame
            frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_file, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            frame_files.append(frame_file)
            
        except Exception as e:
            logger.error(f"Error processing file {wd_file}: {e}")
            continue
    
    # Create GIF from frames
    if frame_files:
        try:
            logger.info(f"Creating GIF animation with {len(frame_files)} frames")
            with imageio.get_writer(output_path, mode='I', duration=interval/1000) as writer:
                for frame_file in frame_files:
                    image = imageio.imread(frame_file)
                    writer.append_data(image)
            
            logger.info(f"Animation saved to {output_path}")
            
            # Clean up temporary files
            for frame_file in frame_files:
                os.remove(frame_file)
            os.rmdir(temp_dir)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating GIF: {e}")
            return None
    else:
        logger.error("No frames were created, cannot generate GIF")
        return None

def find_the_time_step_with_max_flood_extent(files):
    max_extent_count = 0
    max_extent_file = None
    max_extent_file_time = None
    for file in files:
        # Extract the time from the filename
        time_str = file.split('_')[-1].split('.')[0]
        
        innun_data = gdal_asarray(file)
        # Count the number of pixels with water depth > 0
        flood_extent_count = np.sum(innun_data > 0)
        if flood_extent_count > max_extent_count:
            max_extent_count = flood_extent_count
            max_extent_file = file
            max_extent_file_time = time_str
            
    return max_extent_file, max_extent_file_time

def plot_extent_map(flood_file, output_file, title=None):
    logger.info(f"Creating flood extent map from {flood_file}")
    dem_file = os.path.join(CARLISLE_DATA_DIR, "Carlisle_5m.asc")
    
    try:
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            dem_nodata = src.nodata
            profile = src.profile
        
        with rasterio.open(flood_file) as src:
            wd_data = src.read(1)
            wd_nodata = src.nodata

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.7, origin='upper')
        masked_wd = np.ma.masked_where((wd_data == wd_nodata) | (wd_data <= 0.01), wd_data)
        im = ax.imshow(masked_wd, extent=extent, cmap='Blues', alpha=0.8, 
                      vmin=0, vmax=np.max(masked_wd) if not np.ma.is_masked(np.max(masked_wd)) else 3,
                      origin='upper')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Water Depth (m)')
        
        if title is None:
            time_str = Path(flood_file).stem.split('-')[-1]
            title = f'Maximum Flood Extent - Time: {time_str}'
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        ax.grid(True, color='gray', alpha=0.3, linestyle='--')
        flood_area = np.sum(masked_wd > 0.01) * profile['transform'][0] * profile['transform'][4]  # Area in m²
        max_depth = np.max(masked_wd)
        logger.info(f"Flood statistics - Max depth: {max_depth:.2f}m, Area: {flood_area/1000000:.2f}km²")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Flood extent map saved to {output_file}")
        plt.close()
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating flood extent map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_extent_reference():
    # Read the test event data 
    test_event_files = glob.glob(os.path.join(SIMULATION_DATA_DIR, "Run1-*.wd"))
    test_event_files.sort()
    # test_event_files = test_event_files[8:]  # skip the first 8 files
    
    # Find the timestep with the maximum flood extent
    max_extent_file, timestep = find_the_time_step_with_max_flood_extent(test_event_files)
    logger.info(f"Maximum flood extent found in file: {max_extent_file} at timestep: {timestep}")
    
    # Create and save the visualization
    output_file = os.path.join(OUTPUT_DIR, "maximum_flood_extent.png")
    # return plot_extent_map(max_extent_file, output_file)

def plot_extent_prediction(run_id=None, idx="0145"):
    if run_id is None:
        # Use the most recent run if not specified
        runs = sorted(glob.glob(os.path.join(RUN_DIR, "1DCNN_V1", "*")))
        if not runs:
            logger.error("No CNN model runs found")
            return None
        run_id = os.path.basename(runs[-1])
    
    # Locate the model prediction file
    max_extent_file = os.path.join(RUN_DIR, "1DCNN_V1", run_id, "output_maps", f"map_{idx}.wd")
    logger.info(f"Visualising CNN model prediction for {run_id} at timestep {idx}")
    
    # Create custom title
    title = f'CNN Model Predicted Flood Extent (Run: {run_id}, Time: {idx})'
    
    # Create and save the visualization
    output_file = os.path.join(GRAPH_OUTPUT_DIR, "maximum_flood_extent_cnn1d.png")
    return plot_extent_map(max_extent_file, output_file, title)

def plot_extents_on_same_image():
    
    #CNN model run
    runs = sorted(glob.glob(os.path.join(RUN_DIR, "1DCNN_V1", "*")))
    run_id = os.path.basename(runs[-1])
    idx = "0145"
    cnn_extent_file = os.path.join(RUN_DIR, "1DCNN_V1", run_id, "output_maps", f"map_{idx}.wd")
    
    #USRR model run
    usrr_runs = sorted(glob.glob(os.path.join(RUN_DIR, "USSR_CNN1D_COMBINED", "*")))
    usrr_run_id = os.path.basename(usrr_runs[-1])
    usrr_extent_file = os.path.join(RUN_DIR, "USSR_CNN1D_COMBINED", usrr_run_id, f"map_{idx}.wd")
    
    #LISFLOOD run
    lf_extent_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{idx}.wd")
    
    # Check if files exist
    if not os.path.exists(cnn_extent_file) or not os.path.exists(lf_extent_file) or not os.path.exists(usrr_extent_file):
        logger.error(f"One or more extent files don't exist:\nCNN: {cnn_extent_file}\nLISFLOOD: {lf_extent_file}\nUSSR: {usrr_extent_file}")
        return None
    
    logger.info(f"Creating collage of models at timestep {idx}")
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    output_file = os.path.join(GRAPH_OUTPUT_DIR, f"model_comparison_{idx}.png")
    
    try:
        # Create figure with a better title and layout
        fig = plt.figure(figsize=(20, 14))  # Increased height for two rows
        fig.suptitle('Flood Extent Comparison - Multiple Models', fontsize=20, fontweight='bold', y=0.98)
        
        # Create a 2×2 grid layout with the third subplot centered in the second row
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.15, hspace=0.2)
        
        # Create three subplots with new arrangement
        ax1 = fig.add_subplot(gs[0, 0])  # LISFLOOD (top left)
        ax2 = fig.add_subplot(gs[0, 1])  # CNN (top right)
        ax3 = fig.add_subplot(gs[1, :])  # USRR (bottom center, spanning both columns)
        
        # Process DEM data
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            dem_nodata = src.nodata
            
        # Process LISFLOOD map
        with rasterio.open(lf_extent_file) as src:
            lf_data = src.read(1)
            lf_nodata = src.nodata
            
        # Process CNN map
        with rasterio.open(cnn_extent_file) as src:
            cnn_data = src.read(1)
            cnn_nodata = src.nodata
            
        # Process USRR map
        with rasterio.open(usrr_extent_file) as src:
            usrr_data = src.read(1)
            usrr_nodata = src.nodata
        
        # Create masked arrays for all flood extents
        cnn_data[cnn_data < 0.03] = 0
        usrr_data[usrr_data < 0.03] = 0
        masked_lf = np.ma.masked_where((lf_data == lf_nodata) | (lf_data==0), lf_data)
        masked_cnn = np.ma.masked_where((cnn_data == cnn_nodata)| (cnn_data==0), cnn_data)
        masked_usrr = np.ma.masked_where((usrr_data == usrr_nodata) | (usrr_data==0),  usrr_data)
        
        # Set a fixed scale for water depth (0-3 meters)
        max_depth = 3.0  # Fixed scale for water depth
        
        # Create high-contrast water colormap with stronger blues
        water_colors = plt.cm.Blues(np.linspace(0, 1, 256))
        # Make the blues more saturated and darker
        for i in range(len(water_colors)):
            # Increase saturation and value for more vibrant blues
            water_colors[i, 0:3] = np.clip(water_colors[i, 0:3] * 1.3, 0, 1)  # Boost color intensity
        
        water_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('enhanced_blues', water_colors)
        
        # Use terrain colormap for DEM
        dem_cmap = plt.cm.Greys_r
        
        # Enhanced background styling with clearer DEM visualization
        # Create a normalized and enhanced DEM for better visibility
        dem_min, dem_max = np.percentile(dem_data, [5, 95])  # Use percentiles to avoid outliers
        normalized_dem = dem_data
        
        # Adjust alpha values for higher contrast
        dem_alpha = 0.7  # Slightly reduced to make water stand out
        water_alpha = 1.0  # Full opacity for water
        
        # Create binary masks for water extent borders
        binary_lf = np.where(~masked_lf.mask, 1, 0)
        binary_cnn = np.where(~masked_cnn.mask, 1, 0)
        binary_usrr = np.where(~masked_usrr.mask, 1, 0)
        
        # Draw LISFLOOD map with enhanced styling
        ax1.imshow(normalized_dem, extent=extent, cmap=dem_cmap, alpha=dem_alpha, origin='upper')
        im1 = ax1.imshow(masked_lf, extent=extent, cmap=water_cmap, alpha=water_alpha, 
                        vmin=0, vmax=3.0, origin='upper')  # Fixed range 0-3
        
        
        
        # Add subplot label and title
        ax1.text(0.05, 0.95, 'A', transform=ax1.transAxes, fontsize=16, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        ax1.set_title('LISFLOOD Simulation', fontsize=14, fontweight='bold', pad=10)
        
        # Draw CNN map with same enhanced styling
        ax2.imshow(normalized_dem, extent=extent,cmap=dem_cmap, alpha=dem_alpha, origin='upper')
        im2 = ax2.imshow(masked_cnn, extent=extent, cmap=water_cmap, alpha=water_alpha, 
                        vmin=0, vmax=3.0, origin='upper')  # Fixed range 0-3
        
        
        
        # Add subplot label and title
        ax2.text(0.05, 0.95, 'B', transform=ax2.transAxes, fontsize=16, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        ax2.set_title('1DCNN Model Prediction', fontsize=14, fontweight='bold', pad=10)
        
        # Draw USRR map with same enhanced styling
        ax3.imshow(normalized_dem, extent=extent, cmap=dem_cmap, alpha=dem_alpha, origin='upper')
        im3 = ax3.imshow(masked_usrr, extent=extent, cmap=water_cmap, alpha=water_alpha, 
                        vmin=0, vmax=3.0, origin='upper')  # Fixed range 0-3
        
        
        # Add subplot label and title
        ax3.text(0.05, 0.95, 'C', transform = ax3.transAxes, fontsize=16, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        ax3.set_title('USRR-1DCNN Model Prediction', fontsize=14, fontweight='bold', pad=10)
        
        # Improve grid styling for all subplots
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3, linestyle='--', color='black')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Add border to each subplot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
        
        # Add a north arrow to all subplots
        for ax in [ax1, ax2, ax3]:
            ax.text(0.95, 0.05, '↑N', transform=ax.transAxes, fontsize=14, 
                    fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
            
            # Add scale bar to each subplot
            scalebar_length_m = 500  # Length in meters
            scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
            scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
            ax.plot([scale_x, scale_x + scalebar_length_m], [scale_y, scale_y], 'k-', linewidth=2)
            ax.text(scale_x + scalebar_length_m/2, scale_y + (extent[3] - extent[2]) * 0.01, 
                    f'{scalebar_length_m}m', ha='center', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Add a single colorbar for water depth with improved styling
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label('Water Depth (m) [0-3m]', fontsize=12, fontweight='bold')
        
        # Adjust layout and save figure
        plt.tight_layout()
        fig.subplots_adjust(top=0.9, right=0.9, wspace=0.1)
        
        # Save the figure
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison map saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating comparison map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def visualise_area_check_map():
    file = os.path.join(GRAPH_OUTPUT_DIR, "area_check.tif")
    output_file = os.path.join(GRAPH_OUTPUT_DIR, "area_check_visualization.png")
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    
    try:
        # Check if file exists
        if not os.path.exists(file):
            logger.error(f"Area check file not found: {file}")
            return None
            
        # Open and read the file
        logger.info(f"Visualizing area check map from: {file}")
        
        # Load DEM data for background
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            dem_nodata = src.nodata
        
        # Load area check data
        with rasterio.open(file) as src:
            data = src.read(1)
            nodata = src.nodata
            
        # Create masked array to handle NoData values
        masked_data = np.ma.masked_where((data == nodata) | (data == 0), data) if nodata else data
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display DEM as background with terrain colormap - increased opacity to 0.9
        ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.9, origin='upper')
        
        # Create custom colormap with white and grey colors - with higher opacity
        colors = [(1, 1, 1, 0), (0.8, 0.8, 0.8, 0.7), (0.4, 0.4, 0.4, 0.8)]  # Transparent, Light Grey, Dark Grey
        cmap_name = 'area_check_cmap'
        cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
        
        # Display the area check data with the custom colormap
        im = ax.imshow(masked_data, extent=extent, cmap=cm, vmin=0, vmax=2, origin='upper')
        
        # Create binary masks for values 1 and 2
        level1_mask = np.where(data == 1, 1, 0)
        level2_mask = np.where(data == 2, 1, 0)
        
        # Add contour lines around the areas with values 1 and 2
        ax.contour(level1_mask, levels=[0.5], colors=['red'], linewidths=0.8,
                  extent=extent, origin='upper')
        ax.contour(level2_mask, levels=[0.5], colors=['darkred'], linewidths=1.2,
                  extent=extent, origin='upper')
                  
        # Find and add grid to each contiguous region of stacked values
        from scipy import ndimage
        
        # Find all connected regions with value 1
        labeled_array1, num_features1 = ndimage.label(level1_mask)
        # Find all connected regions with value 2
        labeled_array2, num_features2 = ndimage.label(level2_mask)
        
        # Function to add grid to a region
        def add_grid_to_region(region_mask, color, linewidth):
            for region_id in range(1, np.max(region_mask) + 1):
                # Get region pixels
                region = (region_mask == region_id)
                if np.sum(region) < 10:  # Skip very small regions
                    continue
                    
                # Find region bounds
                rows, cols = np.where(region)
                min_row, max_row = np.min(rows), np.max(rows)
                min_col, max_col = np.min(cols), np.max(cols)
                
                # Calculate grid cell size
                height = max_row - min_row
                width = max_col - min_col
                
                # Create 9×14 grid within the region
                row_steps = np.linspace(min_row, max_row, 10)  # 9 cells = 10 lines
                col_steps = np.linspace(min_col, max_col, 15)  # 14 cells = 15 lines
                
                # Convert grid to data coordinates
                pixel_height = (extent[3] - extent[2]) / data.shape[0]
                pixel_width = (extent[1] - extent[0]) / data.shape[1]
                
                y_grid = [extent[3] - r * pixel_height for r in row_steps]  # Top to bottom
                x_grid = [extent[0] + c * pixel_width for c in col_steps]  # Left to right
                
                # Draw horizontal grid lines
                for y in y_grid:
                    ax.axhline(y=y, color=color, linestyle='-', alpha=0.4, linewidth=linewidth)
                
                # Draw vertical grid lines
                for x in x_grid:
                    ax.axvline(x=x, color=color, linestyle='-', alpha=0.4, linewidth=linewidth)
        
        # Add grids to level 1 regions (orange)
        add_grid_to_region(labeled_array1, 'orange', 0.5)
        
        # Add grids to level 2 regions (red)
        add_grid_to_region(labeled_array2, 'red', 0.7)
        
        # Add colorbar with custom ticks and labels
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=[0, 1, 2])
        cbar.set_label('Stacked Value')
        cbar.ax.set_yticklabels(['0', '1', '2'])
        
        # Set title and remove axis labels for cleaner visualization
        ax.set_title('Area Verification Map', fontsize=16, fontweight='bold')
        ax.grid(True, color='gray', alpha=0.3, linestyle='--')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Save the figure
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Area check map visualization saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error visualizing area check map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def draw_metrics():
    """Generate and save comparative performance visualizations of different models."""
    metric_file = os.path.join(RUN_DIR, "training_metrics.csv")
    
    try:
        metrics = pd.read_csv(metric_file)
        # Keep only the last row for each model (final performance)
        metrics = metrics.groupby('model').last().reset_index()
        
        # Extract relevant metrics
        model_names = metrics['model'].values
        rmse = metrics['pred_rmse'].values
        inference_times = metrics['pred_time'].values
        params = metrics['trainable_params'].values
        flops = metrics['flops'].values
        neurons = metrics['total_neurons'].values
        logger.info(f"Generating performance comparisons for {len(model_names)} models")
        
        create_performance_plot(inference_times, rmse, model_names, 
                                  'Inference Time (seconds)', 'RMSE (m)', 
                                  'Model Performance Comparison',
                                  'model_performance_comparison.png')
        
        create_performance_plot(params, inference_times, model_names,
                                  'Parameters (Million)', 'Inference Time (seconds)',
                                  'Model Complexity vs Inference Time',
                                  'model_complexity_vs_inference_time.png')
        
        create_performance_plot(flops, params, model_names,
                                  'FLOPs (Billion)', 'Parameters (Million)',
                                  'Model Complexity Comparison',
                                  'model_complexity_comparison.png')
        
        create_performance_plot(neurons, rmse, model_names,
                                  'Total Neurons (Million)', 'RMSE (m)',
                                  'Model Size vs RMSE',
                                  'model_size_vs_rmse.png')
        
        
    except Exception as e:
        logger.error(f"Error generating metrics visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_performance_plot(x_values, y_values, model_names, x_label, y_label, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with color gradient
    scatter = ax.scatter(x_values, y_values, s=100, c=range(len(model_names)), 
                        cmap='viridis', alpha=0.8, edgecolors='black')
    
    # Add model name annotations
    for i, model in enumerate(model_names):
        ax.annotate(model, 
                   (x_values[i], y_values[i]),
                   xytext=(10, 5),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Configure plot
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid and improve aesthetics
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure
    output_file = os.path.join(GRAPH_OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_file}")
    
    plt.close()
  
def plot_study_area(output_filename=None):

    logger.info("Generating study area visualization")
    
    if output_filename is None:
        output_filename = "carlisle_study_area.png"
    
    output_file = os.path.join(GRAPH_OUTPUT_DIR, output_filename)
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    
    try:
        # Load DEM data
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            transform = src.transform
            dem_nodata = src.nodata
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Enhance DEM visualization with terrain colormap and hillshade effect
        # Calculate hillshade for enhanced topographic visualization
        x, y = np.gradient(dem_data)
        slope = np.pi/2 - np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan2(-x, y)
        
        # Light direction and intensity
        azimuth = np.pi/4  # Light from northwest
        altitude = np.pi/4  # 45 degree elevation
        
        # Calculate hillshade
        hillshade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
        hillshade = hillshade * 255  # Scale to 0-255
        
        # Display hillshade with DEM
        ax.imshow(hillshade, extent=extent, cmap='gray', alpha=0.5, origin='upper')
        dem_plot = ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.7, origin='upper')
        
        # Add color bar for elevation
        cbar = fig.colorbar(dem_plot, ax=ax, shrink=0.6)
        cbar.set_label('Elevation (m)', fontsize=12, fontweight='bold')
        
        # Read boundary conditions file to highlight key features
        bci_file = os.path.join(DATA_DIR, "carlisle.bci")
        
        upstream1_points = []
        upstream2_points = []
        upstream3_points = []
        
        try:
            with open(bci_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('//'): 
                        continue
                    
                    parts = line.split()
            
                    if parts[0] == 'P':  # Point source
                        x, y = float(parts[1]), float(parts[2])
                        if parts[4] == 'upstream1':
                            upstream1_points.append((x, y))
                        elif parts[4] == 'upstream2':
                            upstream2_points.append((x, y))
                        elif parts[4] == 'upstream3':
                            upstream3_points.append((x, y))
        except Exception as e:
            logger.warning(f"Could not process boundary conditions: {e}")
        
        # Plot upstream points with distinct markers and colors
        if upstream1_points:
            x, y = zip(*upstream1_points)
            
            # Label the river
            x_ref, y_ref = upstream1_points[0]
            ax.annotate("River Eden", 
                       xy=(x_ref, y_ref),
                       xytext=(30, 30),
                       textcoords="offset points",
                       fontsize=12,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=0.2", 
                                      shrinkA=5, 
                                      shrinkB=5,
                                      mutation_scale=15))  # Shorter arrow with curve
        
        if upstream2_points:
            x, y = zip(*upstream2_points)

            # Label the river
            x_ref, y_ref = upstream2_points[0]
            ax.annotate("River Petteril", 
                       xy=(x_ref, y_ref),
                       xytext=(30, -30),
                       textcoords="offset points",
                       fontsize=12,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=-0.2", 
                                      shrinkA=5, 
                                      shrinkB=5,
                                      mutation_scale=15))  # Shorter arrow with curve
        
        if upstream3_points:
            x, y = zip(*upstream3_points)
            
            # Label the river
            x_ref, y_ref = upstream3_points[0]
            ax.annotate("River Caldew", 
                       xy=(x_ref, y_ref),
                       xytext=(-120, -30),  # Moved further left from -80 to -120
                       textcoords="offset points",
                       fontsize=12,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=0.2", 
                                      shrinkA=5, 
                                      shrinkB=5,
                                      mutation_scale=15))  # Shorter arrow with curve

        # Add specific points of interest (S₁, S₂, S₃) with adjusted positions
        points_of_interest = [
            {"name": "S₁", "easting": 342682, "northing": 557532, "desc": "Upstream1"},
            # Move S₂ and S₃ slightly upward to make them more visible
            {"name": "S₂", "easting": 341362, "northing": 554702 + 50, "desc": "Upstream2"}, # Added +50 to northing
            {"name": "S₃", "easting": 339947, "northing": 554702 + 50, "desc": "Upstream3"}, # Added +50 to northing
        ]
        
        # Use the same color for all points of interest for consistency
        poi_color = '#e41a1c'  # Red color for all points
        
        # Draw these special points LAST to ensure they're on top (zorder controls stacking)
        for i, poi in enumerate(points_of_interest):
            ax.scatter(poi["easting"], poi["northing"], color=poi_color, s=120, 
                      marker='D', edgecolor='black', linewidth=1.5, alpha=0.9, 
                      label=f"{poi['name']}" + (f" ({poi['desc']})" if poi['desc'] else ""),
                      zorder=10)  # Higher zorder brings to front
            
            # Add label with name (also with high zorder)
            # Customize the position for S₁
            if poi["name"] == "S₁":
                xytext = (10, -25)  # Move S₁ label down
            else:
                xytext = (10, 10)  # Default position for other labels
                
            ax.annotate(poi["name"], 
                       xy=(poi["easting"], poi["northing"]),
                       xytext=xytext,
                       textcoords="offset points",
                       fontsize=14,
                       fontweight='bold',
                       color=poi_color,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8),
                       zorder=11)  # Ensure labels are on top
        
        # Add north arrow
        ax.text(0.95, 0.05, '↑N', transform=ax.transAxes, fontsize=16, 
                fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Add scale bar
        scalebar_length_m = 1000  # 1 km
        scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
        scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
        ax.plot([scale_x, scale_x + scalebar_length_m], [scale_y, scale_y], 'k-', linewidth=2)
        ax.text(scale_x + scalebar_length_m/2, scale_y - (extent[3] - extent[2]) * 0.01, 
                f'1 km', ha='center', va='top', 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Option 1: Place legend below the plot
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, 
                  framealpha=0.9, fontsize=10)
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        
        # Save figure with extra space for the legend
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Study area visualization saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating study area visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def plot_model_architecture():
    model = SimpleNet()
    dummy_input = torch.randn(1, 10)
    torch.onnx.export(model, dummy_input, f"{GRAPH_OUTPUT_DIR}/model.onnx", input_names=['input'], output_names=['output'])

