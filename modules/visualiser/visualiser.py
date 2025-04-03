import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR, RUN_DIR, GRAPH_OUTPUT_DIR
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
    test_event_files = test_event_files[8:]  # skip the first 8 files
    
    # Find the timestep with the maximum flood extent
    max_extent_file, timestep = find_the_time_step_with_max_flood_extent(test_event_files)
    logger.info(f"Maximum flood extent found in file: {max_extent_file} at timestep: {timestep}")
    
    # Create and save the visualization
    output_file = os.path.join(OUTPUT_DIR, "maximum_flood_extent.png")
    return plot_extent_map(max_extent_file, output_file)

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