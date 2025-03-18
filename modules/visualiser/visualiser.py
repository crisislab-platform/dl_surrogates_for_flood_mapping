import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from modules.utils.path_util import OUTPUT_DIR,RUN_DIR, DATA_DIR
import logging
import rasterio
from shapely.geometry import shape, Point
import geopandas as gpd

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

