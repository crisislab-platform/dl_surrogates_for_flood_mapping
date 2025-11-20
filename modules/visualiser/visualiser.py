import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from modules.lib.constants import DATA_DIR as DATA_DIR, OUTPUT_DIR, SIMULATION_DATA_DIR, RUN_DIR, GRAPH_OUTPUT_DIR
import logging
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import glob
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
from pathlib import Path
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray


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
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
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


def plot_study_area(output_filename=None, show_spatial_scales=True):

    logger.info("Generating study area visualization with spatial scales")
    
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
            pixel_size = abs(transform[0])  # Get pixel resolution
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 12))
        
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
        
        # Add spatial scale information if requested
        if show_spatial_scales:
            # Add coordinate grid with labeled ticks
            # Calculate nice round numbers for grid spacing
            x_range = extent[1] - extent[0]
            y_range = extent[3] - extent[2]
            
            # Set grid spacing to approximately 1km intervals
            grid_spacing = 1000  # 1km
            
            # Create grid lines
            x_ticks = np.arange(
                np.ceil(extent[0] / grid_spacing) * grid_spacing,
                np.floor(extent[1] / grid_spacing) * grid_spacing + 1,
                grid_spacing
            )
            y_ticks = np.arange(
                np.ceil(extent[2] / grid_spacing) * grid_spacing,
                np.floor(extent[3] / grid_spacing) * grid_spacing + 1,
                grid_spacing
            )
            
            # Set ticks and enable grid
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.grid(True, alpha=0.3, linestyle='--', color='white', linewidth=1)
            
            # Format tick labels to show coordinates in km
            ax.set_xticklabels([f'{int(x/1000)}' for x in x_ticks])
            ax.set_yticklabels([f'{int(y/1000)}' for y in y_ticks])
            ax.set_xlabel('Easting (km)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Northing (km)', fontsize=12, fontweight='bold')
            
            # Add pixel resolution information
            ax.text(0.02, 0.98, f'Resolution: {pixel_size}m', 
                   transform=ax.transAxes, fontsize=15, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                   verticalalignment='top')
            
        
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
                       xytext=(-120, 10),
                       textcoords="offset points",
                       fontsize=12,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=0.2", 
                                      shrinkA=5, 
                                      shrinkB=5,
                                      mutation_scale=15))
        
        if upstream2_points:
            x, y = zip(*upstream2_points)

            # Label the river
            x_ref, y_ref = upstream2_points[0]
            ax.annotate("River Petteril", 
                       xy=(x_ref, y_ref),
                       xytext=(30, 40),
                       textcoords="offset points",
                       fontsize=12,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=-0.2", 
                                      shrinkA=5, 
                                      shrinkB=5,
                                      mutation_scale=15))
        
        if upstream3_points:
            x, y = zip(*upstream3_points)
            
            # Label the river
            x_ref, y_ref = upstream3_points[0]
            ax.annotate("River Caldew", 
                       xy=(x_ref, y_ref),
                       xytext=(-120, 50),
                       textcoords="offset points",
                       fontsize=12,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=0.2", 
                                      shrinkA=5, 
                                      shrinkB=5,
                                      mutation_scale=15))

        # Add specific points of interest (S₁, S₂, S₃) with adjusted positions
        points_of_interest = [
            {"name": "S₁", "easting": 342682, "northing": 557532, "desc": "Upstream1"},
            {"name": "S₂", "easting": 341362, "northing": 554702 + 50, "desc": "Upstream2"},
            {"name": "S₃", "easting": 339947, "northing": 554702 + 50, "desc": "Upstream3"},
        ]
        
        # Use the same color for all points of interest for consistency
        poi_color = '#e41a1c'  # Red color for all points
        
        # Draw these special points LAST to ensure they're on top
        for i, poi in enumerate(points_of_interest):
            ax.scatter(poi["easting"], poi["northing"], color=poi_color, s=120, 
                      marker='D', edgecolor='black', linewidth=1.5, alpha=0.9, 
                      label=f"{poi['name']}" + (f" ({poi['desc']})" if poi['desc'] else ""),
                      zorder=10)
            
            # Add label with name
            if poi["name"] == "S₁":
                xytext = (10, -25)
            else:
                xytext = (10, 10)
                
            ax.annotate(poi["name"], 
                       xy=(poi["easting"], poi["northing"]),
                       xytext=xytext,
                       textcoords="offset points",
                       fontsize=14,
                       fontweight='bold',
                       color=poi_color,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8),
                       zorder=11)
        
        # Add north arrow
        ax.text(0.95, 0.05, '↑N', transform=ax.transAxes, fontsize=16, 
                fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Add multiple scale bars for different scales
        scale_bars = [
            # {"length": 500, "label": "500m", "y_offset": 0.05},
            {"length": 1000, "label": "1km", "y_offset": 0.08}
            # {"length": 2000, "label": "2km", "y_offset": 0.11}
        ]
        
        for scale in scale_bars:
            scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
            scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
            ax.plot([scale_x, scale_x + scale["length"]], [scale_y, scale_y], 
                   'k-', linewidth=3, alpha=0.8)
            ax.text(scale_x + scale["length"]/2, scale_y - (extent[3] - extent[2]) * 0.008, 
                    scale["label"], ha='center', va='top', 
                    fontweight='bold', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Set title with spatial information
        # ax.set_title('Carlisle Study Area - Spatial Domain and Scale Information', 
        #             fontsize=16, fontweight='bold', pad=20)
        
        # Add legend at the top of the plot in two rows with centered second row
        legend = ax.legend(bbox_to_anchor=(0.7, 1.06), loc='lower center', ncol=3, 
                          framealpha=0.9, fontsize=15, columnspacing=2.0)
        
        # Add UK context inset map with real geographical data
        try:
            # Import cartopy for real map data
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.io.img_tiles import OSM, GoogleTiles
            
            # Create inset with cartopy projection
            inset_ax = fig.add_axes([0.02, 0.65, 0.25, 0.25], projection=ccrs.PlateCarree())
            
            # UK bounds for the inset map
            uk_west, uk_east = -8.0, 2.0
            uk_south, uk_north = 49.5, 59.0
            
            # Set the extent to cover the UK
            inset_ax.set_extent([uk_west, uk_east, uk_south, uk_north], crs=ccrs.PlateCarree())
            
            # Add real geographical features
            inset_ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', alpha=0.6)
            inset_ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='black', alpha=0.6)
            inset_ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
            inset_ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
            inset_ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
            
            # Try to add satellite or terrain tiles for more detail
            try:
                # Use OpenStreetMap tiles for a realistic base map
                osm_tiles = OSM()
                inset_ax.add_image(osm_tiles, 6)  # Lower zoom level for UK overview
            except:
                # Fallback to basic features if tiles fail
                inset_ax.add_feature(cfeature.RIVERS, linewidth=0.3, color='blue', alpha=0.6)
            
            # Carlisle coordinates
            carlisle_lon = -2.9336
            carlisle_lat = 54.8951
            
            # Mark Carlisle on inset with location pin marker
            inset_ax.plot(carlisle_lon, carlisle_lat, marker='o', color='red', 
                         markersize=14, markeredgecolor='black', markeredgewidth=1.5, 
                         transform=ccrs.PlateCarree(), zorder=10)
            
            # Add Carlisle annotation next to the marker
            inset_ax.text(carlisle_lon + 0.6, carlisle_lat, 'Carlisle', 
                         transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold',
                         ha='left', va='center', color='black',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                  edgecolor='black', alpha=0.6))
            
            # # Add major UK cities for context
            # major_cities = [
            #     {'name': 'London', 'lon': -0.13, 'lat': 51.51},
            #     {'name': 'Edinburgh', 'lon': -3.19, 'lat': 55.95},
            #     {'name': 'Manchester', 'lon': -2.24, 'lat': 53.48}
            # ]
            
            # for city in major_cities:
            #     inset_ax.plot(city['lon'], city['lat'], marker='s', color='blue', 
            #                  markersize=4, markeredgecolor='black', markeredgewidth=0.5, 
            #                  transform=ccrs.PlateCarree(), zorder=8)
            
            # Add country labels
            # inset_ax.text(-1.5, 52.5, 'ENGLAND', fontsize=7, fontweight='bold', 
            #              ha='center', alpha=0.8, transform=ccrs.PlateCarree())
            # inset_ax.text(-4.0, 56.0, 'SCOTLAND', fontsize=7, fontweight='bold', 
            #              ha='center', alpha=0.8, transform=ccrs.PlateCarree())
            
            # Remove ticks and labels from inset for cleaner look
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            
            # Add border to inset
            for spine in inset_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor('black')
            
            # Add inset title
            # inset_ax.set_title('', fontsize=10, fontweight='bold', pad=5)
            
            logger.info("Added realistic UK context inset with geographical data")
            
        except ImportError:
            logger.warning("Cartopy not available, using simplified UK context inset")
            # 
        except Exception as e:
            logger.warning(f"Could not add UK context inset: {e}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Study area visualization with spatial scales saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating study area visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_study_area_clean(output_filename=None):
    """
    Create a clean study area visualization without labels, text, or colorbar.
    Suitable for presentations or publications where minimal annotation is desired.
    """
    logger.info("Generating clean study area visualization")
    
    if output_filename is None:
        output_filename = "carlisle_study_area_clean.png"
    
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
        
        # Display hillshade with DEM - no colorbar
        ax.imshow(hillshade, extent=extent, cmap='gray', alpha=0.5, origin='upper')
        ax.imshow(dem_data, extent=extent, cmap='terrain', alpha=0.7, origin='upper')
        
        # Remove all ticks, labels, and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Remove axis spines for completely clean look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save figure with minimal whitespace
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
        logger.info(f"Clean study area visualization saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating clean study area visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_study_area_satellite(output_filename=None):
    """
    Create a clean satellite view of the Carlisle study area.
    Uses Google Maps satellite imagery without any additional markers.
    Uses the DEM file extent for consistent coverage across visualizations.
    """
    logger.info("Generating clean satellite view of Carlisle study area")
    
    try:
        # Import cartopy 
        import cartopy.crs as ccrs
        from cartopy.io.img_tiles import GoogleTiles
        from shapely.geometry import box
    except ImportError:
        logger.error("Required packages not found. Please install them with: pip install cartopy")
        return None
    
    if output_filename is None:
        output_filename = "carlisle_study_area_satellite.png"
    
    output_file = os.path.join(GRAPH_OUTPUT_DIR, output_filename)
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    
        
    points_of_interest = [
        {"name": "S₁", "easting": 342682, "northing": 557532, "desc": "Upstream1"},
        {"name": "S₂", "easting": 341362, "northing": 554702 + 50, "desc": "Upstream2"},
        {"name": "S₃", "easting": 339947, "northing": 554702 + 50, "desc": "Upstream3"},
    ]
        
    try:
        # Load DEM data to get the extent
        with rasterio.open(dem_file) as src:
            source_crs = "EPSG:27700"
            # Get the original bounds
            dem_extent = src.bounds
            logger.info(f"DEM bounds ({source_crs}): {dem_extent}")

            # Define the destination CRS (WGS84)
            dest_crs = "EPSG:4326"

            # Use rasterio.warp.transform_bounds to accurately transform the bounds
            # This handles the geometric transformations more precisely
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                source_crs,
                dest_crs,
                dem_extent.left,
                dem_extent.bottom,
                dem_extent.right,
                dem_extent.top
            )

        logger.info(f"DEM bounds (WGS84): [{lon_min}, {lat_min}, {lon_max}, {lat_max}]")
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create tile source for Google satellite imagery
        google_tiles = GoogleTiles(style='satellite')  # Use satellite style explicitly
        
        # Create map with appropriate projection
        ax = plt.axes(projection=google_tiles.crs)
        
        # Add the satellite imagery tiles
        ax.add_image(google_tiles, 14)  # Higher zoom level for more detail
        
        # Set extent to match the DEM bounds
        # Add a small buffer (2%) around the DEM extent for better visualization
        buffer_lon = (lon_max - lon_min) * 0.02
        buffer_lat = (lat_max - lat_min) * 0.02
        
        ax.set_extent([
            lon_min - buffer_lon, 
            lon_max + buffer_lon, 
            lat_min - buffer_lat, 
            lat_max + buffer_lat
        ], crs=ccrs.PlateCarree())
        
        # Add a subtle watermark/attribution in bottom right
        plt.text(0.98, 0.02, '© Google Maps', transform=ax.transAxes,
                fontsize=8, color='white', alpha=0.7, ha='right')
        
        # Remove axis ticks for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a scale bar (approximate)
        # Calculate degrees per km at this latitude
        center_lat = (lat_min + lat_max) / 2
        km_per_degree = 111.32 * np.cos(np.radians(center_lat))
        one_km_in_degrees = 1.0 / km_per_degree
        
        # Add scale bar at bottom left
        scale_lon = lon_min + buffer_lon * 2
        scale_lat = lat_min + buffer_lat * 2
        
        ax.plot([scale_lon, scale_lon + one_km_in_degrees], 
                [scale_lat, scale_lat], 
                'w-', linewidth=3, transform=ccrs.PlateCarree())
        
        ax.text(scale_lon + one_km_in_degrees/2, scale_lat + buffer_lat,
                '1 km', color='white', fontweight='bold', ha='center',
                transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='black', alpha=0.5, boxstyle="round,pad=0.2"))
        
        # Add north arrow (simple text arrow)
        ax.text(0.95, 0.95, '↑\nN', transform=ax.transAxes,
                fontsize=14, color='white', ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle="round,pad=0.3"))
        
        # Save figure
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Satellite view saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating satellite view: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_uk_context_map(output_filename=None):
    """
    Create a map showing Carlisle's location within the UK context.
    This provides geographical context for the study area.
    """
    logger.info("Generating UK context map showing Carlisle location")
    
    try:
        # Import required packages for mapping
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.io.img_tiles import OSM
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error("Required packages not found. Please install them with: pip install cartopy")
        return None
    
    if output_filename is None:
        output_filename = "carlisle_uk_context.png"
    
    output_file = os.path.join(GRAPH_OUTPUT_DIR, output_filename)
    
    try:
        # Carlisle coordinates (approximate city center)
        carlisle_lon = -2.9336  # Longitude
        carlisle_lat = 54.8951  # Latitude
        
        # UK bounds for the map
        uk_west, uk_east = -8.0, 2.0
        uk_south, uk_north = 49.5, 59.0
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 12), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Set the extent to cover the UK
        ax.set_extent([uk_west, uk_east, uk_south, uk_north], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.7)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.7)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, color='blue', alpha=0.6)
        
        # Add a subtle grid
        gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Mark Carlisle with a prominent marker
        ax.plot(carlisle_lon, carlisle_lat, marker='o', color='red', 
               markersize=12, markeredgecolor='black', markeredgewidth=2,
               transform=ccrs.PlateCarree(), zorder=10)
        
        # Add Carlisle label with an arrow
        ax.annotate('Carlisle\nStudy Area', 
                   xy=(carlisle_lon, carlisle_lat),
                   xytext=(carlisle_lon + 1.5, carlisle_lat + 1.0),
                   transform=ccrs.PlateCarree(),
                   fontsize=14,
                   fontweight='bold',
                   ha='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                            edgecolor='red', alpha=0.9),
                   arrowprops=dict(arrowstyle="->", 
                                  connectionstyle="arc3,rad=0.2", 
                                  color='red',
                                  lw=2))
        
        # Add major UK cities for context
        major_cities = [
            {'name': 'London', 'lon': -0.1278, 'lat': 51.5074},
            {'name': 'Manchester', 'lon': -2.2426, 'lat': 53.4808},
            {'name': 'Edinburgh', 'lon': -3.1883, 'lat': 55.9533},
            {'name': 'Glasgow', 'lon': -4.2518, 'lat': 55.8642},
            {'name': 'Newcastle', 'lon': -1.6131, 'lat': 54.9783},
            {'name': 'Birmingham', 'lon': -1.8904, 'lat': 52.4862}
        ]
        
        for city in major_cities:
            ax.plot(city['lon'], city['lat'], marker='s', color='blue', 
                   markersize=6, markeredgecolor='black', markeredgewidth=1,
                   transform=ccrs.PlateCarree(), zorder=8)
            
            # Add city labels (smaller and less prominent than Carlisle)
            ax.text(city['lon'], city['lat'] - 0.3, city['name'],
                   transform=ccrs.PlateCarree(),
                   fontsize=10,
                   ha='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                            alpha=0.7, edgecolor='none'))
        
        # Add country labels
        ax.text(-1.0, 53.0, 'ENGLAND', transform=ccrs.PlateCarree(),
               fontsize=16, fontweight='bold', ha='center', alpha=0.6)
        ax.text(-4.0, 56.5, 'SCOTLAND', transform=ccrs.PlateCarree(),
               fontsize=16, fontweight='bold', ha='center', alpha=0.6)
        ax.text(-3.5, 52.0, 'WALES', transform=ccrs.PlateCarree(),
               fontsize=16, fontweight='bold', ha='center', alpha=0.6)
        
        # Add a compass rose (north arrow)
        ax.text(0.95, 0.95, '↑\nN', transform=ax.transAxes,
               fontsize=16, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor='black', alpha=0.9))
        
        # Add scale bar (approximate)
        scale_bar_length = 1.0  # degrees (approximately 100km at this latitude)
        scale_x = uk_west + 0.5
        scale_y = uk_south + 0.5
        
        ax.plot([scale_x, scale_x + scale_bar_length], 
               [scale_y, scale_y], 
               'k-', linewidth=3, transform=ccrs.PlateCarree())
        
        ax.text(scale_x + scale_bar_length/2, scale_y - 0.3,
               '~100 km', ha='center', va='top', fontweight='bold',
               transform=ccrs.PlateCarree(),
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                        alpha=0.8, edgecolor='black'))
        
        # Create a legend
        legend_elements = [
            mpatches.Patch(color='red', label='Carlisle Study Area'),
            mpatches.Patch(color='blue', label='Major UK Cities'),
            mpatches.Patch(color='lightgray', label='Land'),
            mpatches.Patch(color='lightblue', label='Water Bodies')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98), framealpha=0.9,
                 fontsize=12)
        
        # Set title
        ax.set_title('Carlisle Location within the United Kingdom', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Add inset showing study area detail (optional small box)
        # Create a small rectangle around Carlisle area
        from matplotlib.patches import Rectangle
        study_area_box = Rectangle((carlisle_lon - 0.15, carlisle_lat - 0.1), 
                                  0.3, 0.2,
                                  linewidth=2, edgecolor='red', facecolor='none',
                                  linestyle='--', alpha=0.8,
                                  transform=ccrs.PlateCarree())
        ax.add_patch(study_area_box)
        
        # Save the figure
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"UK context map saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating UK context map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_uk_context_simple(output_filename=None):
    """
    Create a simplified UK context map without cartopy dependency.
    Shows Carlisle's location using basic matplotlib plotting.
    """
    logger.info("Generating simplified UK context map showing Carlisle location")
    
    if output_filename is None:
        output_filename = "carlisle_uk_context_simple.png"
    
    output_file = os.path.join(GRAPH_OUTPUT_DIR, output_filename)
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Simplified UK outline coordinates (rough approximation)
        # These are very simplified coordinates for illustration
        uk_outline_lon = [-5.5, -5.0, -4.5, -3.0, -2.0, -1.0, 0.5, 1.0, 1.5, 
                         1.0, 0.5, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -5.5, -5.5]
        uk_outline_lat = [50.0, 49.8, 50.2, 50.5, 51.0, 51.5, 51.3, 51.8, 52.5,
                         53.5, 54.5, 55.0, 55.8, 56.5, 57.0, 56.0, 54.0, 52.0, 50.0]
        
        # Plot UK outline
        ax.fill(uk_outline_lon, uk_outline_lat, color='lightgray', alpha=0.7, 
               edgecolor='black', linewidth=1.5)
        
        # Carlisle coordinates
        carlisle_lon = -2.9336
        carlisle_lat = 54.8951
        
        # Mark Carlisle
        ax.plot(carlisle_lon, carlisle_lat, marker='o', color='red', 
               markersize=15, markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        # Add Carlisle label
        ax.annotate('Carlisle\nStudy Area', 
                   xy=(carlisle_lon, carlisle_lat),
                   xytext=(carlisle_lon + 1.5, carlisle_lat + 1.0),
                   fontsize=14,
                   fontweight='bold',
                   ha='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                            edgecolor='red', alpha=0.9),
                   arrowprops=dict(arrowstyle="->", 
                                  connectionstyle="arc3,rad=0.2", 
                                  color='red', lw=2))
        
        # Add some major cities for context
        cities = [
            {'name': 'London', 'lon': -0.13, 'lat': 51.51},
            {'name': 'Edinburgh', 'lon': -3.19, 'lat': 55.95},
            {'name': 'Manchester', 'lon': -2.24, 'lat': 53.48}
        ]
        
        for city in cities:
            ax.plot(city['lon'], city['lat'], marker='s', color='blue', 
                   markersize=8, markeredgecolor='black', markeredgewidth=1, zorder=8)
            ax.text(city['lon'], city['lat'] - 0.4, city['name'],
                   fontsize=10, ha='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
        
        # Add country labels
        ax.text(-1.5, 52.5, 'ENGLAND', fontsize=14, fontweight='bold', 
               ha='center', alpha=0.6)
        ax.text(-4.0, 56.0, 'SCOTLAND', fontsize=14, fontweight='bold', 
               ha='center', alpha=0.6)
        
        # Set map properties
        ax.set_xlim(-6, 2)
        ax.set_ylim(49, 59)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add north arrow
        ax.text(0.95, 0.95, '↑\nN', transform=ax.transAxes,
               fontsize=14, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor='black', alpha=0.9))
        
        # Set title
        ax.set_title('Carlisle Location within the United Kingdom', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Create simple legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=12, markeredgecolor='black', label='Carlisle Study Area'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                   markersize=8, markeredgecolor='black', label='Major UK Cities')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=11)
        
        # Save the figure
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Simple UK context map saved to {output_file}")
        plt.close()
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating simple UK context map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None



