from modules.lib.constants import SIMULATION_DATA_DIR, OUTPUT_DIR
import os
from modules.models.srr_lstm.srr.find_starting_points import get_starting_pts_from_maximum_inundation_extent
from modules.models.srr_lstm.srr.sdr_runner import perform_SDR, Traj_search
import time
from modules.models.srr_lstm.srr.SDR_algorithm_filter import Select_Rep_pts, extract_rep_pts
import logging
import numpy as np
from modules.models.srr_lstm.srr.gdal_func import gdal_asarray
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SDRReducer")

class SDRReducer():
    def __init__(self):
        self.dem_asc_file = f"{SIMULATION_DATA_DIR}/Carlisle_5m.asc"
        self.max_inundation_timestep = {
            'event_id': 3,
            'timestep': 94
        }
        self.max_inundation_file = f"{SIMULATION_DATA_DIR}/Run{self.max_inundation_timestep['event_id']}-{(self.max_inundation_timestep['timestep']):04d}.wd"
        self.non_flood_mask_file = f"{SIMULATION_DATA_DIR}/Run3-0008.wd"
        self.stopping_values_dem = [-555, -666]
        self.result_dir = os.path.join(OUTPUT_DIR, "sdr_reduction_results")
        os.makedirs(self.result_dir, exist_ok=True)
        self.sdr_file_thalwegs = os.path.join(self.result_dir, "sdr_results_thalwegs.shp")
        self.sdr_file_mcl = os.path.join(self.result_dir, "sdr_results_mcl.shp")
        self.customised_searching_win_size = 9
        self.initial_run = True
        self.require_rep_thalweg_selection = True
        self.rep_traj_ratio = 1/200
        
        #starting coordinates
        self.starting_coords = None
        self.dem_aggregate_rate=3
        self.dem_aggregate_function=np.nanmean
        
        #Starting point for MCL is the upstream1 boundary. 
        #The coordinates of the point
        self.mcl_starting_point = np.array([342682,  557532]) #northing and easting
        
        # RL selection
        self.resample_rate_mcl = 200
        self.resample_rate_sdr_thalwegs = 0.1
        self.ouput_rls_file_name = os.path.join(self.result_dir, "representative_locations.shp")
    
    def visualise_stopping_categories(self, map):
        # Visualise the stopping categories with exactly three discrete values
        plt.figure(figsize=(12, 8))
        
        # Create custom colormap for the three discrete categories
        from matplotlib.colors import ListedColormap
        import matplotlib.colors as mcolors
        
        # Define specific colors: 0=white, -555=lightblue, -666=yellow
        colors = ['yellow', 'lightblue', 'white']  # -666, -555, 0
        
        # Get unique values to verify we have exactly three categories
        unique_vals = np.unique(map)
        logger.info(f"Unique values in DEM: {unique_vals}")
        
        # Create discrete bounds for the three values
        bounds = [-666.5, -555.5, -0.5, 0.5]
        cmap = ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        plt.imshow(map, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Create colorbar with the three discrete categories
        cbar = plt.colorbar(ticks=[-666, -555, 0], label='Stopping Categories')
        cbar.set_ticklabels(['Downstream Boundary (-666)', 'Non-flood (-555)', 'Normal DEM (0)'])
        
        plt.title('Stopping Categories Visualization')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Save the visualization
        mask_viz_path = os.path.join(self.result_dir, "stopping_categories_visualization.png")
        plt.savefig(mask_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Stopping categories visualization saved to {mask_viz_path}")
        

    def sdr_searching(self):
        starting_pts = get_starting_pts_from_maximum_inundation_extent(self.max_inundation_file, None)
        self.starting_coords = starting_pts
        if self.require_rep_thalweg_selection:
            starttime = time.time()
            output_file_name = f"{self.sdr_file_thalwegs[:-4]}_before_rep_thal_selec.shp"
            perform_SDR(self.result_dir, starting_pts, self.dem_asc_file, output_file_name, self.stopping_values_dem, self.customised_searching_win_size, (not self.initial_run))
            Select_Rep_pts(self.result_dir, self.dem_asc_file, output_file_name).save_rep_trajs_2shp_with_this_ratio(self.sdr_file_thalwegs, self.rep_traj_ratio)
            logger.info(f"SDR search completed in {time.time() - starttime:.2f} seconds. Results saved to {self.sdr_file_thalwegs}")
        else:
            starttime = time.time()
            perform_SDR(self.result_dir, starting_pts, self.dem_asc_file, self.sdr_file_thalwegs, self.stopping_values_dem, self.customised_searching_win_size, (not self.initial_run))
            logger.info(f"SDR search completed. Results saved to {self.sdr_file_thalwegs}")
            
        logger.info(f"example initial point for SDR search: { str(self.starting_coords[0]) } , { str(self.starting_coords[1]) }")
        return 0
    
    def sdr_searching_mcl(self):
        logger.info("Starting SDR MCL search...")
        starttime = time.time()
        starting_points = np.array(self.mcl_starting_point).reshape(1,2)
        stopping_values = [-666]  # Downstream boundary, non-flood, normal DEM
        Traj_search(self.result_dir, starting_points, self.dem_asc_file, stopping_values, self.customised_searching_win_size)\
        .generate_main_river_shp(self.sdr_file_mcl, self.dem_aggregate_rate, self.dem_aggregate_function)
        logger.info(f"SDR MCL search completed in {time.time() - starttime:.2f} s. Results saved to {self.sdr_file_mcl}")

    def sdr_rl(self):
        starttime = time.time()
        extract_rep_pts(self.sdr_file_mcl, self.sdr_file_thalwegs, self.resample_rate_mcl, self.resample_rate_sdr_thalwegs, self.result_dir, self.ouput_rls_file_name)
        # Need a unique identifier for each representative location
        logger.info(f"SDR RL extraction completed in {time.time() - starttime:.2f} seconds. Results saved to {self.ouput_rls_file_name}")