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
    Create a scientific-quality hydrograph plot for Event 1 with all upstream sources on the same plot.
    Designed to meet publication standards with proper formatting and annotations.
    """
    logger.info("Generating scientific-quality upstream hydrograph for Event 1")
    
    # Load Event 1 flow data
    flow_file = os.path.join(CARLISLE_DATA_DIR, "Upstream_Flows_Run1.csv")
    
    if not os.path.exists(flow_file):
        logger.error(f"Event 1 flow file not found: {flow_file}")
        return None
    
    try:
        # Read the flow data
        df = pd.read_csv(flow_file)
        df = df[8:] 
        df = df.reset_index(drop=True)
        # Convert time from seconds to hours
        if "Time" in df.columns:
            df["TimeHours"] = (df["Time"] / 3600) - 2  # Adjust to start from zero hours
            time_col = "TimeHours"
            time_label = "Time (hours)"
        else:
            time_col = "Time"
            time_label = "Time (s)"
        
        # Create output directory
        os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
        
        # Set scientific plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure with specific size for publication (typically 7.5 inches wide for single column)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        
        # Define scientific color palette (colorblind-friendly)
        colors = ['#0072B2', '#D55E00', '#009E73']  # Blue, Orange-Red, Green
        
        # Define line styles for different data series
        line_styles = ['-', '--', '-.']
        
        # Define river names and markers
        upstream_sources = ['Upstream1', 'Upstream2', 'Upstream3']
        labels = ['S₁ (Eden)', 'S₂ (Petteril)', 'S₃ (Caldew)']
        markers = ['o', 's', '^']
        
        # Plot all upstream sources with scientific styling
        max_values = []
        for i, source in enumerate(upstream_sources):
            if source in df.columns:
                # Plot the hydrograph with publication-quality styling
                ax.plot(df[time_col], df[source], 
                        linestyle=line_styles[i],
                        linewidth=2.0, 
                        color=colors[i], 
                        label=labels[i],
                        marker=markers[i],
                        markevery=int(len(df)/10),  # Show markers at intervals
                        markersize=6)
                
                # Calculate and store peak values
                max_value = df[source].max()
                max_time = df.loc[df[source].idxmax(), time_col]
                max_values.append((source, max_time, max_value))
        
        # Set labels with proper scientific formatting and LaTeX rendering
        ax.set_ylabel('Discharge ($m^{3} s^{-1}$)', fontsize=14, fontweight='bold')
        ax.set_xlabel(time_label, fontsize=14, fontweight='bold')
        
        # Add proper tick marks
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        # Set grid style for scientific plots
        ax.grid(True, which='major', linestyle='-', alpha=0.7, color='lightgray')
        ax.grid(True, which='minor', linestyle=':', alpha=0.4, color='lightgray')
        
        # Annotate peak values with scientific notation
        for source, max_time, max_value in max_values:
            i = upstream_sources.index(source)
            ax.annotate(f'$Q_{{max}} = {max_value:.1f}$ $m^{3}/s$',
                        xy=(max_time, max_value),
                        xytext=(10, 10),
                        textcoords='offset points',
                        color=colors[i],
                        fontsize=15,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.8))
        
        # Add legend with scientific styling
        legend = ax.legend(loc='upper right', fontsize=15, framealpha=0.9, 
                  edgecolor='gray', title="Monitoring Points")
        legend.get_title().set_fontweight('bold')
        
        # Add light box around the plot for scientific paper style
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            
        # Set y-axis to start from 0 for scientific accuracy in comparison
        ax.set_ylim(bottom=0)
        
        # Add event information text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        textstr = '\n'.join((
            r'$\bf{Event\ 1\ Characteristics}$',
            f'Duration: {df[time_col].max():.1f} hours',
            f'Peak discharge: {max(m[2] for m in max_values):.1f} $m^3/s$'
        ))
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=15,
                verticalalignment='top', bbox=props)
        
        # Add figure number and caption (publication style)
        # fig.text(0.5, 0.01, 'Hydrograph of input boundary conditions for the three rivers in Event 1.', 
        #         ha='center', fontsize=15, style='italic')
        
        # Apply tight layout with appropriate margins
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        # Save the figure with high resolution required for publication
        output_path = os.path.join(GRAPH_OUTPUT_DIR, "upstream_hydrographs_event1_scientific.png")
        plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
        
        
        logger.info(f"Saved scientific-quality Event 1 hydrograph to {output_path}")
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating scientific hydrograph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
