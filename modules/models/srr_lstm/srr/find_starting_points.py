from modules.models.srr_lstm.srr.gdal_func import rc2coords, gdal_asarray
from modules.models.srr_lstm.srr.nc_func import *
import matplotlib.pyplot as plt
import numpy as np
from modules.lib.constants import GRAPH_OUTPUT_DIR, SIMULATION_DATA_DIR
import os
import logging

logger = logging.getLogger(__name__)

def get_boundary_pts(arr, dry_cell_threshold):
    arr_01 = arr.copy()
    arr_01[arr_01 > 0] = 1
    checking_arrs = [arr_01[0:-2, 0:-2], arr_01[0:-2, 1:-1], arr_01[0:-2, 2:],
                     arr_01[1:-1, 0:-2], arr_01[1:-1, 1:-1], arr_01[1:-1, 2:],
                     arr_01[2:,   0:-2], arr_01[2:,   1:-1], arr_01[2:,   2:]]
    count_arr = checking_arrs[0] + checking_arrs[1] + checking_arrs[2] + checking_arrs[3] + checking_arrs[5] + \
                checking_arrs[6] + checking_arrs[7] + checking_arrs[8]
    count_arr[count_arr >= dry_cell_threshold] = 0   # valid cells: count between 1~3
    target_arr = count_arr * checking_arrs[4]
    target_arr[target_arr != 0] = 1     # located cells for starting with value 1 (others 0)
    target_arr = np.r_[[np.zeros(target_arr.shape[1])], target_arr, [np.zeros(target_arr.shape[1])]]   # add row at beginning & end
    target_arr = np.c_[np.zeros(target_arr.shape[0]), target_arr, np.zeros(target_arr.shape[0])]   # add column at beginning & end
    return target_arr

def get_pts_coords_from_agg(eventfile, maxtime, dry_cell_threshold=4):
    """ get the central points of all the 1 cells in the aggregated raster"""
    if eventfile[-2:] == 'nc':
        ncdata = Dataset(eventfile)
        maxarr = ncarr2gdalarr(ncdata.variables['water_level'][maxtime*4, :, :].data)
        maxarr[maxarr == -999] = np.nan
        arr_agg, agg_trans = maxarr, getNCtransform(ncdata)
    elif eventfile[-3:] == 'tif' or eventfile[-2:] == 'wd':
        arr_agg = gdal_asarray(eventfile)
        agg_trans = gdal.Open(eventfile).GetGeoTransform()
    else:
        raise TypeError('The file type of the file for starting points selection cannot be recognized. Only NetCDF/.nc or GeoTiff/.tif can be used.')
    boundary_pts = get_boundary_pts(arr_agg, dry_cell_threshold)
    idx = np.array(list(np.where(boundary_pts == 1)))
    xy_coords = np.zeros(idx.shape)
    for i in range(idx.shape[1]):
        xy_coords[:, i] = rc2coords(agg_trans, idx[:, i])
    return xy_coords, boundary_pts


def get_starting_pts_from_maximum_inundation_extent(eventfile, maxtime, nanthreshold=4):
    """
    Inputs:
        eventfile: the file directory/filename of the maximum inundation map.
                   Only NetCDF/.nc or GeoTiff/.tif format can be recognized
                   The non-inundated area needs to be marked as nan; other area being any number other than nan.
        maxitime: if .nc file is provided, indicate the layer in the first dimension to use, count starts from 0;
                   If .tif format is used, input 0.
        nanthreshold: default=4, the threshold used to identify boundary grids as described in our paper
    Output:
        a numpy array of shape (N, 2), N number of coordinates (x, y)
    """
    pts,arr = get_pts_coords_from_agg(eventfile, maxtime, nanthreshold)
    visualise_starting_points(arr)
    return np.array(pts).T

def visualise_starting_points(starting_pts):
        dem_asc_file = f"{SIMULATION_DATA_DIR}/Carlisle_5m.asc"
        # Visualise the starting points on the DEM
        plt.figure(figsize=(12, 8))
        
        # Load the DEM data
        dem_data = gdal_asarray(dem_asc_file)
        
        # Plot the DEM
        plt.imshow(dem_data, cmap='terrain', interpolation='nearest', alpha=0.4)
        
        # Plot the starting points map as overlay
        # Create a masked array where 0 values are transparent
        starting_pts_masked = np.ma.masked_where(starting_pts == 0, starting_pts)
        plt.imshow(starting_pts_masked, cmap='Reds', interpolation='nearest', alpha=0.8)
        
        plt.title('Starting Points on DEM')
        plt.xlabel('X Coordinate (Grid)')
        plt.ylabel('Y Coordinate (Grid)')
        plt.colorbar(label='Starting Points')
        
        # Save the visualization
        starting_pts_viz_path = os.path.join(GRAPH_OUTPUT_DIR, "starting_points_visualization.png")
        plt.savefig(starting_pts_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Starting points visualization saved to {starting_pts_viz_path}")