from modules.lib.constants import CARLISLE_DATA_DIR,SIMULATION_DATA_DIR, OUTPUT_DIR, GRAPH_OUTPUT_DIR, RUN_DIR
import os
import logging
import pandas as pd
import glob
from matplotlib import pyplot as plt
import numpy as np
from modules.lib.constants import CNN1D_V1, PICNN1D_V1, USRR_CNN1D_COMBINED, SRR_LSTM_COMBINED, HDL_FM_V1
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Test Event Visualisation")

def plot_depth_predictions_at_points():
    """
    Plots the predicted vs. true flood depths at points of interest.
    Creates one graph per point, with predictions from all available models overlaid.
    """
    
    logger.info("Generating point-centric plots comparing predictions from all models")
    model_names = [CNN1D_V1, PICNN1D_V1, USRR_CNN1D_COMBINED, SRR_LSTM_COMBINED, HDL_FM_V1]
    
    # Create output directory for comparison plots
    output_path = os.path.join(OUTPUT_DIR, "quality_metrics")
    os.makedirs(output_path, exist_ok=True)
    
    poi_path = os.path.join(OUTPUT_DIR, "points_of_interest.csv")
    if not os.path.exists(poi_path):
        logger.error(f"Points of interest file not found at {poi_path}")
        return
        
    # Load points of interest
    poi_df = pd.read_csv(poi_path)
    point_ids = poi_df['Point_ID'].unique()
    
    # Dictionary to store prediction data by model
    model_predictions = {}
    
    # First, collect all available model predictions
    for model_name in model_names:
        predictions_path = os.path.join(RUN_DIR, model_name, "predictions_at_points_final.csv")
        if not os.path.exists(predictions_path):
            logger.warning(f"Predictions file not found at {predictions_path}, skipping model {model_name}")
            continue
            
        try:
            pred_df = pd.read_csv(predictions_path)
            # Convert timestep to hours (assuming each timestep is 15 minutes)
            pred_df['TimeHours'] = pred_df['Timestep'] / 4
            model_predictions[model_name] = pred_df
            logger.info(f"Loaded {len(pred_df)} prediction records for model {model_name}")
        except Exception as e:
            logger.error(f"Error loading predictions for {model_name}: {str(e)}")
    
    if not model_predictions:
        logger.error("No model predictions found. Cannot create comparison plots.")
        return
    
    # Create summary comparison plot for all models
    plot_summary_comparison(model_predictions, poi_df)
    
    # Use the same vibrant colors for each model as in performance_and_footprint.py
    model_colors = {
        CNN1D_V1: "#00BFA5",        # Bright teal
        PICNN1D_V1: "#E91E63",      # Bright pink
        USRR_CNN1D_COMBINED: "#2196F3",  # Bright blue
        SRR_LSTM_COMBINED: "#FFC107",  # Amber/gold
        HDL_FM_V1: "#9C27B0",       # Bright purple
        'True': 'black'  # Ground truth
    }
    
    # Define line styles for better differentiation
    model_styles = {
        CNN1D_V1: '-',
        PICNN1D_V1: '--',
        USRR_CNN1D_COMBINED: '-.',
        SRR_LSTM_COMBINED: ':',
        HDL_FM_V1: (0, (3, 1, 1, 1)),  # Custom dash pattern
        'True': '-'  # Ground truth
    }
    
    # Create nicer display names for models
    model_display_names = {
        CNN1D_V1: '1DCNN',
        PICNN1D_V1: 'PICNN',
        USRR_CNN1D_COMBINED: 'USRR-CNN',
        SRR_LSTM_COMBINED: 'SRR-LSTM',
        HDL_FM_V1: 'HDL-FM',
        'True': 'Ground Truth'
    }
    
    # For each point, create a graph showing all models
    for point_id in point_ids:
        # Get point metadata
        point_meta = poi_df[poi_df['Point_ID'] == point_id].iloc[0]
        point_label = point_meta['Label']
        point_elev = point_meta['Elevation_m']
        
        # Create figure
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        # First, plot the ground truth from the first available model
        # (ground truth should be the same in all model files)
        first_model = list(model_predictions.keys())[0]
        point_data = model_predictions[first_model][model_predictions[first_model]['Point_ID'] == point_id]
        point_data = point_data.sort_values('TimeHours')
        
        # Plot ground truth
        true_line, = ax.plot(point_data['TimeHours'], point_data['True_Depth_m'], 
                    linewidth=3.0, color=model_colors['True'], 
                    linestyle=model_styles['True'],
                    label=f"{model_display_names['True']}")
        
        # Store metrics for legend
        legend_entries = [true_line]
        legend_labels = [model_display_names['True']]
        
        # Plot each model's predictions
        for model_name, pred_df in model_predictions.items():
            if point_id not in pred_df['Point_ID'].unique():
                logger.warning(f"Point {point_id} not found in predictions for model {model_name}")
                continue
                
            # Extract data for this point
            point_data = pred_df[pred_df['Point_ID'] == point_id]
            point_data = point_data.sort_values('TimeHours')
            
            # Calculate error metrics
            rmse = np.sqrt(np.mean((point_data['True_Depth_m'] - point_data['Predicted_Depth_m'])**2))
            
            # Plot predicted depth
            model_line, = ax.plot(point_data['TimeHours'], point_data['Predicted_Depth_m'], 
                        linewidth=2.0, color=model_colors[model_name],
                        linestyle=model_styles[model_name],
                        label=f"{model_display_names[model_name]} (RMSE: {rmse:.3f}m)")
                        
            # Add to legend entries
            legend_entries.append(model_line)
            legend_labels.append(f"{model_display_names[model_name]} (RMSE: {rmse:.3f}m)")
            
            # Find and annotate peak
            if point_data['Predicted_Depth_m'].std() > 0.01:
                pred_max_idx = point_data['Predicted_Depth_m'].idxmax()
                pred_peak_hour = point_data.loc[pred_max_idx, 'TimeHours']
                pred_peak_depth = point_data.loc[pred_max_idx, 'Predicted_Depth_m']
                
                # Only annotate if there's a significant peak
                if pred_peak_depth > 0.05:
                    ax.scatter([pred_peak_hour], [pred_peak_depth], color=model_colors[model_name], s=50, zorder=5)
        
        # Find and annotate ground truth peak
        if point_data['True_Depth_m'].std() > 0.01:
            true_max_idx = point_data['True_Depth_m'].idxmax()
            true_peak_hour = point_data.loc[true_max_idx, 'TimeHours']
            true_peak_depth = point_data.loc[true_max_idx, 'True_Depth_m']
            
            if true_peak_depth > 0.05:
                ax.scatter([true_peak_hour], [true_peak_depth], color=model_colors['True'], s=80, zorder=6)
                ax.annotate(f'Peak: {true_peak_depth:.2f}m',
                        xy=(true_peak_hour, true_peak_depth),
                        xytext=(10, 10),
                        textcoords='offset points',
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))
        
        # Format the plot
        ax.set_title(f"Point {point_id}: {point_label} (Elevation: {point_elev:.2f}m)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Water Depth (m)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Format x-axis ticks
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        
        def hour_formatter(x, pos):
            return f"{int(x)}h" if x == int(x) else ""
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(hour_formatter))
        
        # Set y-axis limits to show more detail if values are small
        if point_data['True_Depth_m'].max() < 0.5:
            upper_limit = max(0.5, point_data['True_Depth_m'].max() * 1.2)
            ax.set_ylim(-0.05, upper_limit)
        
        # Add legend with error metrics
        ax.legend(handles=legend_entries, labels=legend_labels, 
                loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add a watermark or data source note
        fig.text(0.98, 0.02, 'Carlisle Flood Analysis', 
                fontsize=8, color='gray', ha='right', va='bottom', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        output_file = os.path.join(output_path, f"point_{point_id}_model_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot for point {point_id} to {output_file}")
        plt.close(fig)
        
        
    #for HDL-FM create a plot comparing all points
    if HDL_FM_V1 in model_predictions:
        create_hdl_fm_multi_point_plot(model_predictions[HDL_FM_V1], poi_df, output_path)
    

    logger.info(f"Completed generating point-centric comparison plots for {len(point_ids)} points")

def plot_summary_comparison(model_predictions, poi_df):
    """
    Creates a summary plot with each model as a row and each point as a column
    Args:
        model_predictions: Dictionary of model predictions DataFrames
        poi_df: DataFrame containing point of interest metadata
    """
    try:
        # Get available models and points
        available_models = list(model_predictions.keys())
        n_models = len(available_models)
        
        # Get unique points from the first model
        first_model = list(model_predictions.keys())[0]
        point_ids = sorted(model_predictions[first_model]['Point_ID'].unique())
        n_points = len(point_ids)
        
        # Create figure with grid: rows = models, columns = points
        fig_width = max(20, 4 * n_points)  # At least 4 inches per point, minimum 20
        fig_height = max(12, 3 * n_models)  # At least 3 inches per model, minimum 12
        
        fig, axes = plt.subplots(n_models, n_points, figsize=(fig_width, fig_height))
        
        # Ensure axes is always 2D
        if n_models == 1:
            axes = axes.reshape(1, -1)
        if n_points == 1:
            axes = axes.reshape(-1, 1)
        
        # Create nicer display names for models
        model_display_names = {
            CNN1D_V1: '1DCNN',
            PICNN1D_V1: 'PI1DCNN',
            USRR_CNN1D_COMBINED: 'USRR-1DCNN',
            SRR_LSTM_COMBINED: 'SRR-LSTM',
            HDL_FM_V1: 'HDL-FM'
        }
        
        # Calculate global y-axis limits across all models for consistency
        all_max_values = []
        for model_name, pred_df in model_predictions.items():
            max_true = pred_df['True_Depth_m'].max()
            max_pred = pred_df['Predicted_Depth_m'].max()
            all_max_values.extend([max_true, max_pred])
        
        global_y_max = max(all_max_values) * 1.1 if all_max_values else 4.0
        global_y_min = -0.05
        
        # Plot each model-point combination
        for model_idx, (model_name, pred_df) in enumerate(model_predictions.items()):
            display_name = model_display_names.get(model_name, model_name)
            
            for point_idx, point_id in enumerate(point_ids):
                ax = axes[model_idx, point_idx]
                
                # Filter data for this point
                point_data = pred_df[pred_df['Point_ID'] == point_id]
                
                if len(point_data) == 0:
                    # No data for this point in this model
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12, color='red')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Get point metadata
                point_meta = poi_df[poi_df['Point_ID'] == point_id]
                if len(point_meta) == 0:
                    point_label = f"Point {point_id}"
                    point_elev = "Unknown"
                else:
                    point_meta = point_meta.iloc[0]
                    point_label = point_meta['Label']
                    point_elev = f"{point_meta['Elevation_m']:.1f}m"
                
                # Sort by time
                point_data = point_data.sort_values('TimeHours')
                
                # Plot true and predicted lines
                ax.plot(point_data['TimeHours'], point_data['True_Depth_m'],
                       linewidth=2.5, color='black', alpha=0.8, label='Truth')
                ax.plot(point_data['TimeHours'], point_data['Predicted_Depth_m'],
                       linewidth=2.0, linestyle='--', color='red', alpha=0.8, label='Prediction')
                
                # Calculate RMSE for this point
                rmse = np.sqrt(np.mean((point_data['True_Depth_m'] - point_data['Predicted_Depth_m'])**2))
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # Set consistent y-axis limits
                ax.set_ylim(global_y_min, global_y_max)
                
                # Format x-axis
                import matplotlib.ticker as ticker
                ax.xaxis.set_major_locator(ticker.MultipleLocator(12))  # Every 12 hours
                
                def hour_formatter(x, pos):
                    return f"{int(x)}h" if x == int(x) else ""
                
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(hour_formatter))
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Set titles and labels
                if model_idx == 0:  # Top row - add point titles
                    ax.set_title(f"{point_label}\n({point_elev})", fontsize=11, fontweight='bold', pad=10)
                
                if point_idx == 0:  # Left column - add model labels
                    ax.set_ylabel(f"{display_name}\nDepth (m)", fontsize=11, fontweight='bold')
                else:
                    ax.set_ylabel("")
                
                if model_idx == n_models - 1:  # Bottom row - add x-labels
                    ax.set_xlabel("Time (hours)", fontsize=10)
                else:
                    ax.set_xlabel("")
                
                # Add RMSE annotation
                ax.text(0.02, 0.98, f'RMSE: {rmse:.3f}m', transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # Remove x-tick labels for all but bottom row
                if model_idx < n_models - 1:
                    ax.set_xticklabels([])
                
                # Remove y-tick labels for all but left column  
                if point_idx > 0:
                    ax.set_yticklabels([])
        
        # Add overall legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2.5, label='Ground Truth'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Model Prediction')
        ]
        
        fig.legend(handles=legend_elements, 
                  loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                  ncol=2, fontsize=12, frameon=True)
        
        # Add main title
        # fig.suptitle('Model Performance Comparison Across All Points', 
        #             fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout with proper margins to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, 
                           hspace=0.4, wspace=0.3)  # Increased hspace and wspace for better spacing
        
        # Save figure
        output_path = os.path.join(GRAPH_OUTPUT_DIR, "model_point_grid_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model-point grid comparison plot to {output_path}")
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating grid comparison plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

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
                f"Max Error: {max(valid_error):.3f}m"
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

def create_error_boxplot():
    """
    Creates a clean, professional box plot showing error distributions for each model.
    """
    logger.info("Creating clean error distribution box plot for all models")
    
    try:
        model_names = [CNN1D_V1, PICNN1D_V1, USRR_CNN1D_COMBINED, SRR_LSTM_COMBINED, HDL_FM_V1]
        model_display_names = {
            CNN1D_V1: '1DCNN',
            PICNN1D_V1: 'PI1DCNN',
            USRR_CNN1D_COMBINED: 'USRR-1DCNN',
            SRR_LSTM_COMBINED: 'SRR-LSTM',
            HDL_FM_V1: 'HDL-FM'
        }
        
        # Collect error data for all models
        error_data = []
        labels = []
        model_stats = {}
        timestep = 136
        
        # Load ground truth
        ref_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{timestep:04d}.wd")
        if not os.path.exists(ref_file):
            logger.error(f"Reference file not found: {ref_file}")
            return
        
        with rasterio.open(ref_file) as src:
            truth_data = src.read(1)
            nodata_value = src.nodata
            # if depth <0.3 set to 0
            truth_data[truth_data < 0.3] = 0.0
        
        for model_name in model_names:
            output_maps_dir = os.path.join(RUN_DIR, "output_maps", model_name)
            pred_file = os.path.join(output_maps_dir, f"map_{timestep:04d}.wd")
            
            if not os.path.exists(pred_file):
                logger.warning(f"Prediction file not found: {pred_file}")
                continue
            
            # Load prediction data
            with rasterio.open(pred_file) as src:
                pred_data = src.read(1)
            

            abs_errors = np.abs(pred_data - truth_data)
            # Ensure it's 1D and flatten if needed
            abs_errors = abs_errors.flatten()
            error_data.append(abs_errors)
            
            display_name = model_display_names.get(model_name, model_name)
            labels.append(display_name)
            pred_data = pred_data.flatten()
            truth_data_flatten = truth_data.flatten();
            
            # Calculate key statistics - fix RMSE calculation
            rmse = np.sqrt(np.mean((pred_data - truth_data_flatten) **2))
            mae = np.mean(abs_errors)
            median_ae = np.median(abs_errors)
            
            model_stats[display_name] = {
                'rmse': rmse,
                'mae': mae,
                'median': median_ae,
                'n_samples': len(abs_errors)
            }
            
            logger.info(f"{display_name}: RMSE={rmse:.3f}m, MAE={mae:.3f}m, Median={median_ae:.3f}m, N={len(abs_errors):,}")
        
        if not error_data:
            logger.error("No error data available for box plot")
            return
        
        # Create clean, professional figure - single plot only
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Main box plot
        bp = ax1.boxplot(error_data, labels=labels, patch_artist=True, 
                        showmeans=True, notch=True,
                        medianprops=dict(color='black', linewidth=2.5),
                        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='darkred', markersize=8),
                        flierprops=dict(marker='o', markersize=3, alpha=0.6, markeredgecolor='none'),
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Use a clean, professional color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(error_data)]
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # Set fixed y-axis range for better visualization
        ax1.set_ylim(0, 1.5)
        
        # Format main plot
        ax1.set_ylabel('Absolute Error (m)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        
        # Add sample size annotations
        for i, (label, stats) in enumerate(model_stats.items()):
            y_pos = 1.4  # Fixed position near top of y-axis
            ax1.text(i+1, y_pos, f'n={stats["n_samples"]:,}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add legend for box plot elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2.5, label='Median'),
            Line2D([0], [0], marker='D', color='red', linewidth=0, markersize=8, label='Mean'),
            Line2D([0], [0], color='gray', linewidth=1.5, label='IQR (25th-75th)')
        ]
        
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)
        
        plt.tight_layout()
        
        output_path = os.path.join(GRAPH_OUTPUT_DIR, "model_error_boxplot_clean.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Clean error box plot saved to {output_path}")
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating clean box plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def vizualise_test_event():
    # plot_upstream_hydrographs()
    # plot_flood_depth()
    # plot_depth_predictions_at_points()
    plot_flood_maps()
    # plot_flood_extent_maps()  # Add flood extent confusion matrix maps
    # create_error_boxplot()  # Add the box plot function# Add individual model maps generation

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
        point_labels = ["Lowest" if p == 0 else "Highest" if p == 100 else f"{p}th precentile" for p in target_percentiles]
        
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

def plot_flood_maps():
    """
    Creates visualizations of flood maps for all available models:
    1. Grid layout with prediction maps for all models
    2. Grid layout with error maps for all models
    
    Each model gets its own row in the grid for easy comparison.
    """
    # List of all models to process
    model_names = [CNN1D_V1, PICNN1D_V1, USRR_CNN1D_COMBINED, SRR_LSTM_COMBINED, HDL_FM_V1]
    
    # Create nicer display names for models
    model_display_names = {
        CNN1D_V1: '1DCNN',
        PICNN1D_V1: 'PI1DCNN',
        USRR_CNN1D_COMBINED: 'USRR-1DCNN',
        SRR_LSTM_COMBINED: 'SRR-LSTM',
        HDL_FM_V1: 'HDL-FM'
    }
    
    # Timestep to use for comparison
    idx = "0145"
    alt_idx = "0136"  # Alternative timestep if primary isn't available
    
    # Reference LISFLOOD run (ground truth)
    lf_extent_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{idx}.wd")
    if not os.path.exists(lf_extent_file):
        logger.error(f"Ground truth file doesn't exist: {lf_extent_file}")
        return None
    
    # Load the reference LISFLOOD data
    with rasterio.open(lf_extent_file) as src:
        truth_data = src.read(1)
        truth_nodata = src.nodata
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    # Load DEM data for background
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    with rasterio.open(dem_file) as src:
        dem_data = src.read(1)
        dem_nodata = src.nodata
    
    # Create colormap for water depth visualization
    water_colors = plt.cm.Blues(np.linspace(0, 1, 256))
    for i in range(len(water_colors)):
        water_colors[i, 0:3] = np.clip(water_colors[i, 0:3] * 1.3, 0, 1)
    water_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('enhanced_blues', water_colors)
    
    # Create diverging colormap for error visualization
    error_cmap = 'coolwarm'  # Red for over-prediction, blue for under-prediction
    
    # Set DEM visualization properties
    dem_cmap = plt.cm.Greys_r
    dem_alpha = 0.7
    water_alpha = 1.0
    
    # Dictionary to collect model data
    model_data = {}
    available_models = []
    
    # Load prediction data for each model
    for model_name in model_names:
        # Try to find the prediction file
        maps_dir = os.path.join(RUN_DIR, "output_maps", model_name)
        primary_map_path = os.path.join(maps_dir, f"map_{idx}.wd")
        alt_map_path = os.path.join(maps_dir, f"map_{alt_idx}.wd")
        
        model_map_path = None
        if os.path.exists(primary_map_path):
            model_map_path = primary_map_path
            used_idx = idx
        elif os.path.exists(alt_map_path):
            model_map_path = alt_map_path
            used_idx = alt_idx
            logger.info(f"Using alternative timestep for {model_name}")
        
        if model_map_path:
            try:
                # Load the prediction data
                with rasterio.open(model_map_path) as src:
                    pred_data = src.read(1)
                    pred_nodata = src.nodata
                
                # Create cleaned prediction data (values < 0.3 are considered dry)
                pred_data_clean = pred_data.copy()
                pred_data_clean[pred_data_clean < 0.3] = 0
                
                # Calculate error (prediction - reference)
                error_data = pred_data_clean - truth_data
                
                # Create masked arrays for visualization
                masked_pred = np.ma.masked_where((pred_data_clean == pred_nodata) | (pred_data_clean == 0), pred_data_clean)
                masked_error = np.ma.masked_where((error_data == pred_nodata), error_data)
                
                # Calculate RMSE
                wet_mask = (truth_data > 0.01) | (pred_data_clean > 0.01)
                valid_mask = (truth_data != truth_nodata) & (pred_data != pred_nodata) & wet_mask
                if np.any(valid_mask):
                    rmse = np.sqrt(np.mean((pred_data[valid_mask] - truth_data[valid_mask]) ** 2))
                else:
                    rmse = 0
                
                # Store data in dictionary
                model_data[model_name] = {
                    'pred_data': pred_data_clean,
                    'error_data': error_data,
                    'masked_pred': masked_pred,
                    'masked_error': masked_error,
                    'rmse': rmse,
                    'timestep': used_idx
                }
                available_models.append(model_name)
                logger.info(f"Loaded prediction data for {model_name}, RMSE: {rmse:.3f}m")
            except Exception as e:
                logger.error(f"Error processing {model_name}: {str(e)}")
        else:
            logger.warning(f"No prediction data found for {model_name}")
    
    if not available_models:
        logger.error("No model prediction data could be loaded")
        return None
    
    logger.info(f"Creating flood map comparison at timestep {idx} for {len(available_models)} models")
    
    # Define output filenames
    output_dir = os.path.join(OUTPUT_DIR, "quality_metrics")
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, f"all_models_predictions_{idx}.png")
    errors_file = os.path.join(output_dir, f"all_models_errors_{idx}.png")
    output_file = os.path.join(output_dir, f"flood_map_error_comparison_{idx}.png")
    
    # Create masked array for reference truth data (once, for comparison to all models)
    masked_truth = np.ma.masked_where((truth_data == truth_nodata) | (truth_data < 0.01), truth_data)
    
    # Calculate global error range for consistent colormaps across all models
    all_error_values = []
    for model in available_models:
        all_error_values.extend(model_data[model]['masked_error'].compressed())
    if all_error_values:
        p95 = np.percentile(np.abs(all_error_values), 95)
        global_error_max = min(p95 * 1.5, max(abs(np.nanmin(all_error_values)), abs(np.nanmax(all_error_values))))
    else:
        global_error_max = 1.0
    
    try:
        # Create a single figure with predictions and errors side by side
        n_models = len(available_models)
        # Calculate figure size - more compact and scientific
        fig_width = 16  # Fixed width for consistency
        fig_height = 12  # Reduced row height for more compact layout
        fig_combined = plt.figure(figsize=(fig_width, fig_height))
    
        # Generate individual model maps
        for model in available_models:
            model_info = model_data[model]
            display_name = model_display_names.get(model, model)
            output_file = os.path.join(output_dir, f"{model}_maps.png")
            
            # Create side-by-side prediction and error maps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left plot: Prediction map
            ax1.imshow(dem_data, extent=extent, cmap=dem_cmap, alpha=dem_alpha, origin='upper')
            im1 = ax1.imshow(model_info['masked_pred'], extent=extent, cmap=water_cmap, alpha=water_alpha, 
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
            
            # Right plot: Error map
            ax2.imshow(dem_data, extent=extent, cmap=dem_cmap, alpha=0.15, origin='upper')
            im2 = ax2.imshow(model_info['masked_error'], extent=extent, cmap=error_cmap,
                            vmin=-global_error_max, vmax=global_error_max, alpha=1.0, origin='upper')
            ax2.set_title(f'{display_name} - Error (RMSE: {model_info["rmse"]:.3f}m)', fontsize=14, fontweight='bold')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Calculate and display error statistics
            error_compressed = model_info['masked_error'].compressed()
            if len(error_compressed) > 0:
                over_pred = np.sum(error_compressed > 0.1) / len(error_compressed) * 100
                under_pred = np.sum(error_compressed < -0.1) / len(error_compressed) * 100
                
                stats_text = (f"Over-prediction: {over_pred:.1f}%\n"
                             f"Under-prediction: {under_pred:.1f}%")
                
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
            
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
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        error_cbar_file = os.path.join(output_dir, "error_colorbar.png")
        plt.savefig(error_cbar_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved error colorbar to {error_cbar_file}")
        plt.close()
        
        # Generate reference map
        logger.info("Generating reference (LISFLOOD) map")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(dem_data, extent=extent, cmap=dem_cmap, alpha=dem_alpha, origin='upper')
        ax.imshow(masked_truth, extent=extent, cmap=water_cmap, alpha=water_alpha, 
                  vmin=0, vmax=3.0, origin='upper')
        
        ax.set_title('Reference (LISFLOOD)', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add scale bar and north arrow
        ax.text(0.95, 0.05, '↑N', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
        
        scalebar_length_m = 500
        scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
        scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
        ax.plot([scale_x, scale_x + scalebar_length_m], [scale_y, scale_y], 'k-', linewidth=2)
        ax.text(scale_x + scalebar_length_m/2, scale_y + (extent[3] - extent[2]) * 0.01, 
               f'{scalebar_length_m}m', ha='center', va='bottom', 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        plt.tight_layout()
        ref_file = os.path.join(output_dir, f"reference_lisflood_{idx}.png")
        plt.savefig(ref_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved reference map to {ref_file}")
        plt.close()
        
        logger.info(f"Generated maps for {len(available_models)} models, 1 reference map, and 2 colorbar images")
        return output_dir
            
    except Exception as e:
        logger.error(f"Error generating individual model maps: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_hdl_fm_multi_point_plot(hdl_fm_predictions, poi_df, output_path):
    """
    Creates a single plot showing HDL-FM predictions across all points of interest
    Args:
        hdl_fm_predictions: DataFrame containing HDL-FM model predictions
        poi_df: DataFrame containing point of interest metadata
        output_path: Directory to save the output plot
    """
    logger.info("Creating multi-point comparison plot for HDL-FM model")
    
    # Get unique point IDs
    point_ids = sorted(hdl_fm_predictions['Point_ID'].unique())
    if len(point_ids) == 0:
        logger.warning("No points found for HDL-FM model")
        return
    
    # Create figure with adequate size
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get elevations for labeling but use distinct colors
    elevations = []
    for point_id in point_ids:
        point_meta = poi_df[poi_df['Point_ID'] == point_id]
        if len(point_meta) > 0:
            elevations.append(point_meta.iloc[0]['Elevation_m'])
        else:
            elevations.append(0)
    
    # Use distinct colors from a vibrant palette for better differentiation
    distinct_colors = [
        "#2196F3",  # Bright blue
        "#E91E63",  # Bright pink
        "#FFC107",  # Amber/gold
        "#00BFA5",  # Bright teal
        "#9C27B0",  # Bright purple
        "#FF5722",  # Deep orange
        "#4CAF50",  # Green
        "#3F51B5",  # Indigo
        "#FF9800",  # Orange
        "#795548",  # Brown
        "#607D8B",  # Blue grey
        "#9E9E9E",  # Grey
    ]
    
    # Ensure we have enough colors by cycling if necessary
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(point_ids))]
    
    # Line styles to alternate between points
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # Plot predictions for each point
    truth_lines = []
    pred_lines = []
    
    for i, point_id in enumerate(point_ids):
        # Filter data for this point
        point_data = hdl_fm_predictions[hdl_fm_predictions['Point_ID'] == point_id]
        if len(point_data) == 0:
            continue
        
        # Sort by time
        point_data = point_data.sort_values('TimeHours')
        
        # Get point metadata
        point_meta = poi_df[poi_df['Point_ID'] == point_id]
        if len(point_meta) > 0:
            point_meta = point_meta.iloc[0]
            point_label = point_meta['Label']
            point_elev = f"{point_meta['Elevation_m']:.1f}m"
        else:
            point_label = f"Point {point_id}"
            point_elev = "Unknown"
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((point_data['True_Depth_m'] - point_data['Predicted_Depth_m'])**2))
        
        # Plot ground truth
        true_line, = ax.plot(point_data['TimeHours'], point_data['True_Depth_m'],
                          color=colors[i], linewidth=2.5, linestyle='-',
                          label=f"{point_id}: True ({point_elev})")
        
        # Plot prediction
        pred_line, = ax.plot(point_data['TimeHours'], point_data['Predicted_Depth_m'],
                          color=colors[i], linewidth=2.0, linestyle=line_styles[i % len(line_styles)],
                          alpha=0.7, label=f"{point_id}: HDL-FM (RMSE: {rmse:.3f}m)")
        
        truth_lines.append(true_line)
        pred_lines.append(pred_line)
        
        # Find and annotate peaks if significant
        if point_data['True_Depth_m'].max() > 0.1:
            true_max_idx = point_data['True_Depth_m'].idxmax()
            true_peak_hour = point_data.loc[true_max_idx, 'TimeHours']
            true_peak_depth = point_data.loc[true_max_idx, 'True_Depth_m']
            
            ax.scatter(true_peak_hour, true_peak_depth, 
                     color=colors[i], marker='o', s=60, zorder=10)
            
            # Add annotation if there's enough space
            if i % 2 == 0:  # Alternate annotation positions
                ax.annotate(f'P{i+1}: {true_peak_depth:.2f}m',
                          xy=(true_peak_hour, true_peak_depth),
                          xytext=(10, 10 + (i % 3) * 10),  # Vary vertical offset
                          textcoords='offset points',
                          fontsize=9,
                          fontweight='bold',
                          color=colors[i],
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], alpha=0.8),
                          arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=colors[i]))
    
    # First create legend for true values
    legend1 = ax.legend(handles=truth_lines, 
                      loc='upper left', 
                      title="Ground Truth",
                      fontsize=18,  # Increased from 9 to 12
                      title_fontsize=18,  # Added title font size
                      framealpha=0.95,   # Increased opacity
                      borderpad=1.0)  # Add padding
    
    # Add the legend for predictions
    ax.add_artist(legend1)
    ax.legend(handles=pred_lines, 
            loc='upper right',
            title="HDL-FM Predictions",
            fontsize=18,  # Increased from 9 to 12
            title_fontsize=18,  # Added title font size
            framealpha=0.95,  # Increased opacity
            borderpad=1.0)  # Add padding
    
    # Customize plot
    ax.set_title("HDL-FM Model Performance Across All Monitoring Points", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (hours)", fontsize=15)
    ax.set_ylabel("Water Depth (m)", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Format x-axis ticks
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
    
    def hour_formatter(x, pos):
        return f"{int(x)}h" if x == int(x) else ""
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(hour_formatter))
    
    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for explanation text
    output_file = os.path.join(output_path, "hdl_fm_all_points_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved HDL-FM multi-point comparison plot to {output_file}")
    plt.close(fig)



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
    model_names = [CNN1D_V1, PICNN1D_V1, USRR_CNN1D_COMBINED, SRR_LSTM_COMBINED, HDL_FM_V1]
    
    # Create nicer display names for models
    model_display_names = {
        CNN1D_V1: '1DCNN',
        PICNN1D_V1: 'PI1DCNN',
        USRR_CNN1D_COMBINED: 'USRR-1DCNN',
        HDL_FM_V1: 'HDL-FM'
    }
    
    # Timestep to use for comparison
    idx = "0145"
    alt_idx = "0136"  # Alternative timestep if primary isn't available
    
    # Define the flood threshold
    flood_threshold = 0.3
    
    # Reference LISFLOOD run (ground truth)
    lf_extent_file = os.path.join(SIMULATION_DATA_DIR, f"Run1-{idx}.wd")
    if not os.path.exists(lf_extent_file):
        logger.error(f"Ground truth file doesn't exist: {lf_extent_file}")
        return None
    
    # Load the reference LISFLOOD data
    with rasterio.open(lf_extent_file) as src:
        truth_data = src.read(1)
        truth_nodata = src.nodata
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    # Load DEM data for background
    dem_file = os.path.join(SIMULATION_DATA_DIR, "Carlisle_5m.asc")
    with rasterio.open(dem_file) as src:
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
        primary_map_path = os.path.join(maps_dir, f"map_{idx}.wd")
        alt_map_path = os.path.join(maps_dir, f"map_{alt_idx}.wd")
        
        model_map_path = None
        if os.path.exists(primary_map_path):
            model_map_path = primary_map_path
            used_idx = idx
        elif os.path.exists(alt_map_path):
            model_map_path = alt_map_path
            used_idx = alt_idx
            logger.info(f"Using alternative timestep for {model_name}")
        
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
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot DEM as background (very light)
            dem_masked = np.ma.masked_equal(dem_data, dem_nodata)
            ax.imshow(dem_masked, extent=extent, cmap='terrain', alpha=0.8, origin='upper')
            
            # Plot confusion matrix map
            im = ax.imshow(confusion_map, extent=extent, cmap=confusion_cmap, 
                          alpha=0.8, origin='upper', vmin=0, vmax=3)
       
            
            # Set title with statistics
            ax.set_title(f'({chr(97 + len(available_models))}) {display_name} - Flood Extent Confusion Matrix', 
                        fontsize=14)
        
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
            stats_text = (f"Hits(A): {tp_count:,}\n"
                         f"Overpredications/Flase Alarms(B): {fp_count:,}\n"
                         f"Misses/Underprediction(C): {fn_count:,}\n"
                         f"Correct Dry(D): {tn_count:,}")
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            plt.tight_layout()
            
            # Save individual model confusion matrix map
            output_file = os.path.join(output_dir, f"{display_name}_confusion_matrix.png")
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
    fig, ax = plt.subplots(figsize=(8, 4))
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
    
    ax.legend(handles=legend_elements, loc='center', fontsize=14, 
             title=f"Flood Extent Classification (Threshold: {flood_threshold}m)",
             title_fontsize=16, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    legend_file = os.path.join(output_dir, "confusion_matrix_legend.png")
    plt.savefig(legend_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved confusion matrix legend to {legend_file}")
    plt.close()
    
    # Create summary statistics table and save as image
    create_confusion_matrix_summary(model_stats, output_dir)
    
    logger.info(f"Generated confusion matrix maps for {len(available_models)} models")
    logger.info("Summary statistics:")
    
    return output_dir

def create_confusion_matrix_summary(model_stats, output_dir):
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
    csv_file = os.path.join(output_dir, "confusion_matrix_summary.csv")
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
    