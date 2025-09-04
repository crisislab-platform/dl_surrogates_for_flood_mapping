from modules.models.srr_lstm.srr.gdal_func import coords2rc, ogr, gdal
import numpy as np
import fiona, os, math
from modules.models.srr_lstm.srr.sdr_algorithm_functions import dem_checking, directory_checking, points_along_line
from modules.models.srr_lstm.srr.sdr_algorithm_functions import perform_DUPLEX_to_pts_for_sdr_rl, save_to_shp_points_for_sdr_rl
from time import time
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import matplotlib.colors as colors
from matplotlib.patches import Patch
from modules.models.srr_lstm.srr.gdal_func import gdal_asarray
from modules.lib.constants import SIMULATION_DATA_DIR

import logging
logger = logging.getLogger(__name__)    

class Select_Rep_pts:
    """
    Select representative points from the trajectories saved in SHP file
    """

    def __init__(self, result_work_dir, demfile, traj_shp, rep_traj_ratio=1 / 200, stoping_categories=(-555, -666)):
        self.work_dir = result_work_dir
        self.demfile = demfile
        self.dem_dataset = gdal.Open(demfile)
        self.non_flood_mask_file =  f"{SIMULATION_DATA_DIR}/Run3-0008.wd"
        self.dem_transform = self.dem_dataset.GetGeoTransform()
        self.demarr = self.dem_dataset.GetRasterBand(1).ReadAsArray()
        self.preprocess_dem(stoping_categories)
        self.drainage_map = np.zeros(self.demarr.shape)
        self.stopping_vals = stoping_categories
        self.countline = [0]
        self.trajfile = traj_shp
        self.representative_trajs_ratio = rep_traj_ratio
    
    def preprocess_dem(self, stopping_categories):
        #Add -555 and -666 to the demarr 
        non_flood_mask = gdal_asarray(self.non_flood_mask_file)
        demarr = self.demarr.copy()
        if  -555 in stopping_categories:
            # Set non-flooded areas to -555
            demarr[non_flood_mask > 0] = -555
        
        if -777 in stopping_categories:
            # Set non-flooded areas to -555
            demarr[non_flood_mask <= 0] = -777
        
        if -666 in stopping_categories:
            # Find the leftmost column with flood values > 0 since downstream boundary is not given in Kabir Study 
            for row in range(non_flood_mask.shape[0]):
                row_data = non_flood_mask[row, 0:5]  # Get first 5 columns of this row
            
                for col in range(row_data.shape[0]):  # Iterate through columns in row_data
                    if row_data[col] > 0:
                        demarr[row, col] = -666
                        
        self.demarr = demarr

    def read_shp_to_trajs(self, shpfile, need_failed_trajs=False):
        trajs = dict()
        failed_pts = list()
        with fiona.open(shpfile) as copy_shp:
            for feature in copy_shp:
                geom = feature['geometry']['coordinates']
                trajs[geom[0]] = geom[1:]
                r, c = coords2rc(self.dem_transform, trajs[geom[0]][-1])
                if self.is_not_stopping_grids(self.demarr[r, c]):
                    failed_pts.append(geom[0])  # [(x,y), (x,y), ...]
        self.countline[0] = len(list(trajs.keys())) - len(failed_pts)
        print(f'The total number of initially successful trajectories: {self.countline[0]}/{len(list(trajs.keys()))}')
        if need_failed_trajs:
            return trajs, failed_pts
        else:
            return trajs

    def is_not_stopping_grids(self, grid_val):
        if np.isnan(grid_val):
            return False
        for stop_val in self.stopping_vals:
            if grid_val == stop_val:
                return False
        return True

    def save_to_shp_lines(self, trajs, outfile):
        shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outfile):
            shpDriver.DeleteDataSource(outfile)
        outDataSource = shpDriver.CreateDataSource(outfile)
        outLayer = outDataSource.CreateLayer(outfile, geom_type=ogr.wkbMultiLineString)
        featureDefn = outLayer.GetLayerDefn()
        for key in trajs.keys():
            multiline = ogr.Geometry(ogr.wkbMultiLineString)
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(key[0], key[1])
            for pts in trajs[key]:
                line.AddPoint(pts[0], pts[1])
            multiline.AddGeometry(line)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(multiline)
            outLayer.CreateFeature(outFeature)
            del multiline, line, outFeature

    def save_to_shp_points(self, pts, outfile):
        shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outfile):
            shpDriver.DeleteDataSource(outfile)
        outDataSource = shpDriver.CreateDataSource(outfile)
        outLayer = outDataSource.CreateLayer(outfile, geom_type=ogr.wkbPoint)
        featureDefn = outLayer.GetLayerDefn()
        for coords in pts:
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(coords[0], coords[1])
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(point)
            outLayer.CreateFeature(outFeature)
            del point, outFeature

        # trajectory filter

    def trajs_filtering(self, trajs):
        """
        output:
            traj_categories = {cate_val0:traj_cate_0, cate_val1:traj_cate_1}
                traj_cates = {coords_of_last_pt_of_sig_traj:traj_group_0, }
                    traj_group: dict((st_coords):[line_ls], ...,
                                    'sig_traj':[line_ls], ...)
        """
        traj_categories = dict()
        for category_val in self.stopping_vals:
            traj_categories[category_val] = dict()
        for st_coords in trajs.keys():
            if len(trajs[st_coords]) <= 20:  # delete trajs with less than 20 points
                continue
            end_val = self.demarr[coords2rc(self.dem_transform, trajs[st_coords][-1])]
            if np.isnan(end_val):  # delete trajs with nan at end
                continue
            else:
                for category_val in self.stopping_vals:
                    if float(end_val) == float(category_val):  # fall into one category (-555 / -666)
                        if trajs[st_coords][-1] in traj_categories[category_val].keys():
                            traj_categories[category_val][trajs[st_coords][-1]][st_coords] = trajs[st_coords]
                        # elif min_dist_met:  # the distance between the current last point and an existing key pt is less than a threshold ?
                        #     # append ? swap ? need ?
                        else:
                            traj_categories[category_val][trajs[st_coords][-1]] = dict()
                            traj_categories[category_val][trajs[st_coords][-1]][st_coords] = trajs[st_coords]
                        break
        # summarizing each group in main category (ending in the river)
        main_category = self.stopping_vals[0]  # -555, river
        main_cate_dict = traj_categories[main_category]
        traj_categories[main_category] = self.trajs_filtering_summary_main_cate(main_cate_dict)
        return traj_categories

    def trajs_filtering_summary_main_cate(self, main_cate_dict):
        for last_pt_coords in main_cate_dict.keys():  # each group; ending with the same grid
            # summarise significant trajs, their length, coverage extent/area, near river dem
            group_dict = main_cate_dict[last_pt_coords]
            length_ls = [len(line_ls) for line_ls in list(group_dict.values())]
            longest_idx = length_ls.index(max(length_ls))
            # choose representative starting pts from each 'group' with a chosing ratio
            potential_pts = list(group_dict.keys())
            potential_pts.remove(list(group_dict.keys())[longest_idx])
            if len(potential_pts) - 2 > 0:
                choosed_pts = [list(group_dict.keys())[longest_idx], last_pt_coords]
                main_cate_dict[last_pt_coords]['rep_starting_pts'] = self.choose_pts_DUPLEX(potential_pts,
                                                                                            choosed_pts,
                                                                                            self.representative_trajs_ratio)
                main_cate_dict[last_pt_coords]['rep_starting_pts'].append(list(group_dict.keys())[longest_idx])
            else:
                main_cate_dict[last_pt_coords]['rep_starting_pts'] = [list(group_dict.keys())[longest_idx]]
            main_cate_dict[last_pt_coords]['significant_traj'] = group_dict[list(group_dict.keys())[longest_idx]]
            main_cate_dict[last_pt_coords]['near_river_dem'] = self.demarr[coords2rc(
                self.dem_transform, main_cate_dict[last_pt_coords]['significant_traj'][-2])]
        return main_cate_dict

    def choose_pts_DUPLEX(self, potential_pts, starting_pts, ratio_of_choose):
        # choose points based on coordinates by DUPLEX method
        no_of_chosen_pts = math.ceil(len(potential_pts) * ratio_of_choose)
        potential_pts = np.array(potential_pts)
        if starting_pts is None:
            starting_pts = np.array([])
            all_pts = potential_pts
            dist_m = dist.cdist(potential_pts, potential_pts, 'sqeuclidean')
            r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
            dist_m[dist_m == 0] = np.nan
            chosen_pts_idxs = -np.ones(no_of_chosen_pts, dtype=int)
            chosen_pts_idxs[[0, 1]] = [r1, r2]
            remaining_idxs = np.array(range(potential_pts.shape[0]))
            remaining_idxs[[r1, r2]] = -1
            for i in range(no_of_chosen_pts - 2):
                r_of_remain = np.nanargmax(np.nanmin(
                    dist_m[:, chosen_pts_idxs[:i + 2]][remaining_idxs[remaining_idxs != -1]], axis=1))
                target_idx = remaining_idxs[remaining_idxs != -1][r_of_remain]
                chosen_pts_idxs[i + 2] = target_idx
                remaining_idxs[target_idx] = -1
        else:
            starting_pts = np.array(starting_pts)
            all_pts = np.concatenate((starting_pts, potential_pts), axis=0)
            dist_m = dist.cdist(all_pts, all_pts, 'sqeuclidean')
            dist_m[dist_m == 0] = np.nan
            chosen_pts_idxs = -np.ones(no_of_chosen_pts, dtype=int)
            chosen_pts_idxs = np.concatenate((np.array(range(starting_pts.shape[0])), chosen_pts_idxs), axis=0)
            remaining_idxs = np.array(range(all_pts.shape[0]))
            remaining_idxs[np.array(range(starting_pts.shape[0]))] = -1
            for i in range(no_of_chosen_pts):
                r_of_remain = np.nanargmax(np.nanmin(
                    dist_m[:, chosen_pts_idxs[:i + starting_pts.shape[0]]][remaining_idxs[remaining_idxs != -1]],
                    axis=1))
                target_idx = remaining_idxs[remaining_idxs != -1][r_of_remain]
                chosen_pts_idxs[i + starting_pts.shape[0]] = target_idx
                remaining_idxs[target_idx] = -1
        chosen_pts = all_pts[chosen_pts_idxs[starting_pts.shape[0]:], :]
        chosen_pts = [tuple(item) for item in chosen_pts]
        return chosen_pts

    def extract_rep_trajs(self, traj_category):
        rep_trajs = {}
        main_cate_dict = traj_category[self.stopping_vals[0]]
        for last_pt_coords in main_cate_dict.keys():
            key_ls = main_cate_dict[last_pt_coords]['rep_starting_pts']
            for ky in key_ls:
                rep_trajs[ky] = main_cate_dict[last_pt_coords][ky]
        return rep_trajs

    def save_rep_trajs_2shp_with_this_ratio(self, outshp, filter_ratio):
        self.representative_trajs_ratio = filter_ratio
        trajs = self.read_shp_to_trajs(self.trajfile)
        logger.info(f"Total number of trajectories read: {len(trajs)}")
        filtered_trajs_category = self.trajs_filtering(trajs)
        logger.info(f"Number of filtered trajectories: {len(filtered_trajs_category)}")
        self.save_to_shp_lines(self.extract_rep_trajs(filtered_trajs_category), outshp)
        self.visualize_drainage(outshp, trajs)  
        return 0

    def visualize_drainage(self, shp_file, trajs):
        """
        Visualize the drainage network shapefile and save the visualization as a PNG file.
        
        Args:
            shp_file (str): Path to the shapefile
            trajs (dict, optional): Trajectory dictionary if already loaded
        """
        try:
            logger.info("Plotting drainage network visualization")
            # Create output filename for the visualization
            vis_output = os.path.splitext(shp_file)[0] + '_visualization.png'
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create a DEM background for context
            dem_masked = np.ma.masked_where(self.demarr >= 900, self.demarr)  # Mask out high values
            dem_masked = np.ma.masked_where(dem_masked == -555, dem_masked)  # Mask non-flood areas
            dem_masked = np.ma.masked_where(dem_masked == -666, dem_masked)  # Mask boundary areas
            
            # Get the extent for the plot from the DEM transform
            x_min = self.dem_transform[0]
            y_max = self.dem_transform[3]
            x_max = x_min + self.dem_transform[1] * self.demarr.shape[1]
            y_min = y_max + self.dem_transform[5] * self.demarr.shape[0]
            
            # Plot DEM with a colormap
            cmap = plt.cm.terrain.copy()
            cmap.set_bad('white', 1.0)
            dem_plot = ax.imshow(dem_masked, extent=[x_min, x_max, y_min, y_max], 
                                 cmap=cmap, alpha=0.7)
            plt.colorbar(dem_plot, ax=ax, label='Elevation')
            
            # Plot each trajectory line with a different color for better visualization
            colors = plt.cm.jet(np.linspace(0, 1, len(trajs)))
            for i, start_pt in enumerate(trajs):
                line_coords = [start_pt] + trajs[start_pt]
                xs, ys = zip(*line_coords)
                ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.0, alpha=0.7)
            
            
            # Add title and legend
            ax.set_title('Drainage Network Visualization', fontsize=14)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            # Create legend
            legend_elements = [
                Patch(facecolor='blue', edgecolor='blue', label='Drainage Network'),
                Patch(facecolor='green', edgecolor='green', label='Starting Points'),
                Patch(facecolor='red', edgecolor='red', label='Drainage Points')
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(vis_output, dpi=300)
            plt.close()
            
            logger.info(f"Drainage network visualization saved to {vis_output}")
        except Exception as e:
            logger.error(f"Error creating drainage visualization: {str(e)}")


def extract_rep_pts(mcl_file, sdr_thalwegs_file, resample_rate_mcl, resample_rate_sdr_thalwegs,
                    results_dir, outshp):
    trajs = dict()
    with fiona.open(sdr_thalwegs_file) as copy_shp:
        for feature in copy_shp:
            geom = feature['geometry']['coordinates']
            trajs[geom[0]] = geom[1:]
    last_pt_ls = list(set([trajs[c_k][-1] for c_k in trajs.keys()]))
    trajs = dict()
    with fiona.open(mcl_file) as copy_shp:
        for feature in copy_shp:
            geom = feature['geometry']['coordinates']
            trajs[geom[0]] = geom[1:]
    river_line_ls = list(trajs.values())[0]
    river_pts = points_along_line(river_line_ls, resample_rate_mcl, resample_rate_mcl / 2, resample_rate_mcl / 2)
    selected_pts = perform_DUPLEX_to_pts_for_sdr_rl(last_pt_ls, resample_rate_sdr_thalwegs) + river_pts
    save_to_shp_points_for_sdr_rl(selected_pts, outshp)
    plot_representative_points(selected_pts, river_pts, results_dir)
    print(f"Selected totally {len(selected_pts)} RLs, with {len(river_pts)} along MCL")

def plot_representative_points(selected_pts, river_pts, results_dir):
    """
    Plot the representative points and save the visualization.
    
    Args:
        selected_pts (list): List of selected representative points.
        river_pts (list): List of points along the river.
        results_dir (str): Directory to save the visualization.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from modules.models.srr_lstm.srr.gdal_func import gdal, gdal_asarray
        import os
        from matplotlib.colors import LightSource
        import matplotlib.patches as mpatches
        
        logger.info("Plotting representative points with DEM background")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        dem_file = f"{SIMULATION_DATA_DIR}/Carlisle_5m.asc"
        # If DEM file is found, use it as background
        if dem_file and os.path.exists(dem_file):
            dem_dataset = gdal.Open(dem_file)
            dem_transform = dem_dataset.GetGeoTransform()
            dem_array = dem_dataset.GetRasterBand(1).ReadAsArray()
            
            # Mask out extreme values for better visualization
            dem_masked = np.ma.masked_where(dem_array >= 900, dem_array)
            dem_masked = np.ma.masked_where(dem_masked <= -100, dem_masked)
            
            # Get the extent for the plot from the DEM transform
            x_min = dem_transform[0]
            y_max = dem_transform[3]
            x_max = x_min + dem_transform[1] * dem_array.shape[1]
            y_min = y_max + dem_transform[5] * dem_array.shape[0]
            
            # Create a shaded relief with hillshading for better visualization
            ls = LightSource(azdeg=315, altdeg=45)
            cmap = plt.cm.terrain
            dem_plot = ax.imshow(dem_masked, extent=[x_min, x_max, y_min, y_max], 
                                cmap=cmap, alpha=0.7)
            plt.colorbar(dem_plot, ax=ax, label='Elevation (m)')
            
            # Set the axis limits to the DEM extent
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
        else:
            logger.warning("DEM file not found. Plotting points without background.")
        
        # Extract x and y coordinates from points
        selected_xs, selected_ys = zip(*selected_pts) if selected_pts else ([], [])
        river_xs, river_ys = zip(*river_pts) if river_pts else ([], [])
        
        # Plot the selected points (non-river points)
        non_river_pts = [pt for pt in selected_pts if pt not in river_pts]
        if non_river_pts:
            non_river_xs, non_river_ys = zip(*non_river_pts)
            ax.scatter(non_river_xs, non_river_ys, color='red', marker='o', s=50, 
                      edgecolor='black', label='Drainage Points')
        
        # Plot river points with a different style
        if river_pts:
            ax.scatter(river_xs, river_ys, color='blue', marker='s', s=50, 
                      edgecolor='black', label='River Points')
        
        # Add title and labels
        ax.set_title('Representative Points for Drainage Network', fontsize=14)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend
        ax.legend(loc='best')
        
        # Save the figure
        output_path = os.path.join(results_dir, 'representative_points_visualization.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Representative points visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating representative points visualization: {str(e)}")


