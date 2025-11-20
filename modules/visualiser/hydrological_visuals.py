import os
import pandas as pd
import numpy as np
import glob
import logging
import matplotlib.pyplot as plt
from modules.lib.constants import DATA_DIR, OUTPUT_DIR, GRAPH_OUTPUT_DIR

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
