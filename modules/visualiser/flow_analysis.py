import os
import pandas as pd
import numpy as np
import glob
import logging
import matplotlib.pyplot as plt
from modules.lib.constants import CARLISLE_DATA_DIR, OUTPUT_DIR, GRAPH_OUTPUT_DIR

logger = logging.getLogger("Flow Analysis")

def plot_upstream_hydrographs():
    
    logger.info("Generating upstream hydrograph plots")
    flow_file_pattern = os.path.join(CARLISLE_DATA_DIR, "Upstream_Flows_Run*.csv")
    flow_files = glob.glob(flow_file_pattern)
    
    if not flow_files:
        logger.error(f"No upstream flow files found matching pattern: {flow_file_pattern}")
        return None
    
    # Create data structure to hold flow data by event
    flow_data = {}
    
    # Load all flow data
    for flow_file in flow_files:
        try:
            # Extract event ID from the filename
            event_id = os.path.basename(flow_file).replace("Upstream_Flows_Run", "").replace(".csv", "")
            
            # Read the flow data
            df = pd.read_csv(flow_file)
            
            # Convert time from seconds to hours for better readability
            if "Time" in df.columns:
                df["TimeHours"] = df["Time"] / 3600
            
            # Store the dataframe
            flow_data[event_id] = df
            
        except Exception as e:
            logger.error(f"Error loading flow data from {flow_file}: {e}")
    
    if not flow_data:
        logger.error("Failed to load any flow data")
        return None
    
    # Create output directory
    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
    
    output_files = []
    
    # Create a separate plot for each upstream source (Upstream1, Upstream2, Upstream3)
    upstream_sources = ['Upstream1', 'Upstream2', 'Upstream3']
    
    # For each upstream source, create one plot showing all events
    for source in upstream_sources:
        # Create a figure for this upstream source
        plt.figure(figsize=(12, 8))
        
        # Plot each event on the same figure with different colors
        colors = plt.cm.tab10.colors  # Use a colormap for distinguishing events
        
        # Keep track of legend entries
        legend_entries = []
        
        # Plot each event
        for i, (event_id, df) in enumerate(sorted(flow_data.items())):
            color = colors[i % len(colors)]  # Cycle through colors if more events than colors
            
            # Check if the source column exists
            if source in df.columns:
                # Plot the hydrograph with a unique color for this event
                time_col = 'TimeHours' if 'TimeHours' in df.columns else 'Time'
                line, = plt.plot(df[time_col], df[source], linewidth=2, color=color, 
                               label=f'Event {event_id}')
                
                legend_entries.append(line)
                
                # Find and highlight peak
                max_value = df[source].max()
                max_time = df.loc[df[source].idxmax(), time_col]
                plt.scatter([max_time], [max_value], color=color, s=50)
                
                # Calculate timestep
                if 'Time' in df.columns:
                    timestep = int(df.loc[df[source].idxmax(), 'Time'] / 900) + 1
                    plt.annotate(f'Event {event_id}',
                              xy=(max_time, max_value),
                              xytext=(10, 10),
                              textcoords='offset points',
                              color=color,
                              fontsize=9,
                              arrowprops=dict(arrowstyle='->', color=color),
                              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))
        
        # Set title and labels
        display_name = ""
        if source == 'Upstream1':
            display_name = 'Upstream 1(S₁) - River Eden'
        elif source == 'Upstream2':
            display_name = 'Upstream 2(S₂) - River Caldew'
        elif source == 'Upstream3':
            display_name = 'Upstream 3(S₃) - River Petteril'
        plt.title(f'{display_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Time (hours)' if 'TimeHours' in next(iter(flow_data.values())).columns else 'Time (s)')
        plt.ylabel('Flow Rate (m³/s)')
        plt.grid(True, alpha=0.3)
        
        # Add legend with increased font size
        plt.legend(handles=legend_entries, title="Event IDs", loc='upper right', 
                  prop={'size': 12}, title_fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save individual figure
        output_path = os.path.join(GRAPH_OUTPUT_DIR, f"{source}_all_events.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {display_name} hydrograph to {output_path}")
        output_files.append(output_path)
        plt.close()
    
    # Create a combined figure with all three plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    # fig.suptitle('Upstream Flow Hydrographs - All Sources', fontsize=20, fontweight='bold')
    
    # Label for subplots: A, B, C
    subplot_labels = ['A', 'B', 'C']
    
    for i, source in enumerate(upstream_sources):
        # Add subplot label in the top left corner
        axes[i].text(0.02, 0.95, subplot_labels[i], transform=axes[i].transAxes,
                    fontsize=16, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=3))
        
        display_name = ""
        if source == 'Upstream1':
            display_name = 'S₁ - River Eden'
        elif source == 'Upstream2':
            display_name = 'S₂ - River Petteril'
        elif source == 'Upstream3':
            display_name = 'S₃ - River Caldew'
            
        axes[i].set_title(display_name, fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Flow Rate (m³/s)', fontsize=12, fontweight='bold')
        
        # Only add x-label to the bottom plot
        if i == 2:
            axes[i].set_xlabel('Time (hours)' if 'TimeHours' in next(iter(flow_data.values())).columns else 'Time (s)', 
                             fontsize=12, fontweight='bold')
        
        # Also increase tick label font sizes
        axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        # Plot each event for this upstream source
        legend_entries = []
        for j, (event_id, df) in enumerate(sorted(flow_data.items())):
            color = plt.cm.tab10.colors[j % len(plt.cm.tab10.colors)]
            
            if source in df.columns:
                time_col = 'TimeHours' if 'TimeHours' in df.columns else 'Time'
                line, = axes[i].plot(df[time_col], df[source], linewidth=2, color=color, 
                                    label=f'Event {event_id}')
                legend_entries.append(line)
                
                # Find and highlight peak
                max_value = df[source].max()
                max_time = df.loc[df[source].idxmax(), time_col]
                axes[i].scatter([max_time], [max_value], color=color, s=50)
                
                # Calculate timestep
                if 'Time' in df.columns:
                    timestep = int(df.loc[df[source].idxmax(), 'Time'] / 900) + 1
                    axes[i].annotate(f'Event {event_id}',
                              xy=(max_time, max_value),
                              xytext=(10, 10),
                              textcoords='offset points',
                              color=color,
                              fontsize=9,
                              arrowprops=dict(arrowstyle='->', color=color),
                              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))
        
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(handles=legend_entries, title="Event IDs", loc='upper right', 
                      prop={'size': 12}, title_fontsize=14)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Make room for the title
    
    # Save the combined figure
    combined_output_path = os.path.join(GRAPH_OUTPUT_DIR, "all_upstream_hydrographs.png")
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved combined hydrograph to {combined_output_path}")
    output_files.append(combined_output_path)
    plt.close(fig)
    
    return output_files

def find_peak_inflow_timestep():
    plot_upstream_hydrographs()
    
def plot_hydrograph_clean():
    """
    Create a clean hydrograph plot for Event 1 with all upstream sources on the same plot.
    """
    logger.info("Generating clean upstream hydrograph for Event 1")
    
    # Load Event 1 flow data
    flow_file = os.path.join(CARLISLE_DATA_DIR, "Upstream_Flows_Run1.csv")
    
    if not os.path.exists(flow_file):
        logger.error(f"Event 1 flow file not found: {flow_file}")
        return None
    
    try:
        # Read the flow data
        df = pd.read_csv(flow_file)
        
        # Convert time from seconds to hours
        if "Time" in df.columns:
            df["TimeHours"] = df["Time"] / 3600
            time_col = "TimeHours"
            time_label = "Time (hours)"
        else:
            time_col = "Time"
            time_label = "Time (s)"
        
        # Create output directory
        os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
        
        # Create single figure with clean styling
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Define colors and labels for each upstream source
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        upstream_sources = ['Upstream1', 'Upstream2', 'Upstream3']
        labels = ['S₁', 'S₂', 'S₃']
        
        # Plot all upstream sources on the same axes
        for i, source in enumerate(upstream_sources):
            if source in df.columns:
                # Plot the hydrograph with clean lines and labels
                ax.plot(df[time_col], df[source], linewidth=2.5, color=colors[i], label=labels[i])
                
                # Fill area under the curve with transparency
                ax.fill_between(df[time_col], df[source], alpha=0.2, color=colors[i])
        
        # Clean styling - minimal labels
        ax.set_ylabel('Flow (m³/s)', fontsize=22)
        ax.set_xlabel(time_label, fontsize=22)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=22)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis to start from 0 for better visual comparison
        ax.set_ylim(bottom=0)
        
        # Remove spacing and apply tight layout
        plt.tight_layout()
        
        # Save the clean figure
        output_path = os.path.join(GRAPH_OUTPUT_DIR, "upstream_hydrographs_event1_clean.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved clean Event 1 hydrograph to {output_path}")
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating clean hydrograph: {e}")
        return None
