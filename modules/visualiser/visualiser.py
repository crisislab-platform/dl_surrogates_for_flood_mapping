from modules.lib.constants import DATA_DIR as DATA_DIR
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Visualiser")

from modules.visualiser.hydrological_visuals import find_peak_inflow_timestep
from modules.visualiser.metrics.performance_and_footprint import plot_metrics
from modules.visualiser.quality_metrics import vizualise_test_event
from modules.datamanager.datamanager import create_inundation_map_tensors
from modules.visualiser.metrics.bootstrapping import rmse_stats
from modules.visualiser.flow_analysis import plot_hydrograph_clean 
from modules.visualiser.exploratory_vis import plot_upstream_conditions, plot_study_area, create_flood_animation, plot_extents_on_same_image, plot_extent_prediction, visualise_area_check_map, plot_study_area_satellite, plot_extent_reference

def plot(plot_type):
    if not plot_type:
        logger.error("Plot type is required for plotting")
        exit(1)
    if plot_type == "up_conditions":
        plot_upstream_conditions()
    elif plot_type == "training_history":
        pass
    elif plot_type == "boundary_information":
        plot_study_area()
    elif plot_type == "animation":
        create_flood_animation() 
    elif plot_type == "extent":
        plot_extents_on_same_image()
    elif plot_type == "1dcnn_extent":
        plot_extent_prediction()
    elif plot_type == "area_check":
        visualise_area_check_map()  
    elif plot_type == "plot_metrics":
        plot_metrics()
    elif plot_type == "study_area_clean":
        plot_study_area_satellite()
    elif plot_type == "flow_analysis":
        find_peak_inflow_timestep()
    elif plot_type == "extent_reference":
        plot_extent_reference()
    elif plot_type == "test_event":
        vizualise_test_event()
    elif plot_type == "bootstrap":
        create_inundation_map_tensors()
        rmse_stats()  
    elif plot_type == "hydrograph_clean":
        plot_hydrograph_clean()
    else:
        logger.error(f"Unknown plot type: {plot_type}")
        exit(1)
