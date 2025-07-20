import rasterio
from rasterio import warp, transform
from rasterio.windows import Window
from rasterio.crs import CRS
from rasterio.transform import xy, Affine
import numpy as np
import geopandas
import os
from sys import platform
from modules.lib.constants import CARLISLE_DATA_DIR

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def coords2rc(transform, coords):
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[4]
    col = int((coords[0] - xOrigin) / pixelWidth)
    row = int((yOrigin - coords[1]) / pixelHeight)
    return row, col

def rc2coords(transform_data, rc):
    xOrigin = transform_data[0]
    yOrigin = transform_data[3]
    logger.info(f"rc {rc}")
    pixelWidth = transform_data[1]
    pixelHeight = transform_data[4]
    logger.info(f"transform_data {xOrigin, yOrigin, pixelWidth, pixelHeight}")
    coordX = xOrigin + pixelWidth * (rc[1] + 0.5)
    coordY = yOrigin + pixelHeight * (rc[0] + 0.5)
    return coordX, coordY

def read_shp_point(filename):
    """Read the Point shapefile attributes as a list of xy coordinates"""
    # Keep using geopandas as it's already a good solution
    result = list()
    startPoints = geopandas.read_file(filename)
    for pt in startPoints.geometry:
        result.append((pt.x, pt.y))
    return result


def read_shp_line(filename):
    """read multi-line shapefile as point coords which defining the lines"""
    # Keep using geopandas as it's already a good solution
    output = list()
    lineshp = geopandas.read_file(filename)
    for ln in lineshp.geometry:
        output.append(list(zip(ln.xy[0], ln.xy[1])))
    return output


def gdal_read_prj():
    """Read projection file and return as WKT string"""
    prj_path = f'{CARLISLE_DATA_DIR}/New_500Samples.prj'
    with open(prj_path, 'r') as f:
        prj_txt = f.read()
    
    # Convert ESRI projection string to WKT using rasterio
    crs = CRS.from_wkt(prj_txt)
    return crs.to_wkt()


def gdal_shpprojection(shpfile):
    """Read projection from shapefile"""
    gdf = geopandas.read_file(shpfile)
    return gdf.crs.to_wkt()


def gdal_asarray(rasterfile):
    """Read raster as numpy array"""
    with rasterio.open(rasterfile) as src:
        return src.read(1)


def gdal_transform(rasterfile):
    """Get the geotransform from raster file"""
    with rasterio.open(rasterfile) as src:
        # Return affine transform directly for compatibility with rc2coords
        # but also maintain the GDAL style tuple format for backwards compatibility
        affine = src.transform
        return (affine.c, affine.a, affine.b, affine.f, affine.e, affine.d)


def gdal_writetiff(arr_data, outfile, ras_temp=None, target_transform=None, target_projection=None):
    """Write array to GeoTIFF file"""
    height, width = arr_data.shape
    # Set up metadata from template or provided values
    if ras_temp is not None:
        with rasterio.open(ras_temp) as src:
            transform = src.transform
            crs = src.crs
    elif all([target_transform, target_projection]):
        transform = rasterio.Affine(
            target_transform[1], target_transform[2], target_transform[0],
            target_transform[4], target_transform[5], target_transform[3]
        )
        crs = CRS.from_wkt(target_projection)
    elif target_transform:
        transform = rasterio.Affine(
            target_transform[1], target_transform[2], target_transform[0],
            target_transform[4], target_transform[5], target_transform[3]
        )
        crs = CRS.from_wkt(gdal_read_prj())
    else:
        raise ValueError('Please provide a raster template or target transform (/&projection)!')
    
    # Create the new raster file
    with rasterio.open(
        outfile,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=arr_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr_data, 1)

    return 0

def writewd(filename, arr_data, transform, crs):
    """Write water depth array to .wd file format
    
    Parameters:
    -----------
    filename : str
        Output file path (should end with .wd)
    arr_data : numpy.ndarray
        2D array containing water depth data
    transform : tuple or Affine
        Geotransform information in GDAL format or as rasterio.Affine
    crs : str or CRS
        Coordinate reference system as WKT string or rasterio.crs.CRS object
    
    Returns:
    --------
    int
        0 on success
    """
    height, width = arr_data.shape
    
    # Ensure transform is in the right format
    if isinstance(transform, tuple):
        # Convert from GDAL-style tuple to Affine
        transform = rasterio.Affine(
            transform[1], transform[2], transform[0],
            transform[4], transform[5], transform[3]
        )
    
    # Ensure CRS is in the right format
    if isinstance(crs, str):
        crs = CRS.from_wkt(crs)
    
    # Create the output file
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',  # Use GTiff driver for compatibility
        height=height,
        width=width,
        count=1,
        dtype=arr_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr_data, 1)
    
    logger.info(f"Water depth data written to {filename}")
    return 0

def gdal_asarray_crop_to(raster_file, ref_trans, ref_shape):
    """Read a subset of a raster as numpy array"""
    with rasterio.open(raster_file) as src:
        # Convert ref_trans (GDAL style) to a point
        ulx, xres, xskew, uly, yskew, yres = ref_trans
        pt_x = ulx + xres/2
        pt_y = uly - xres/2  # Using xres here assuming square pixels
        
        # Get row, col for upper left corner
        r0, c0 = coords2rc(src.transform, (pt_x, pt_y))
        
        # Use window to read the exact subset
        window = Window(c0, r0, ref_shape[1], ref_shape[0])
        return src.read(1, window=window)


def gdal_writeasc(filename, arr_data, rastemp):
    """Write array to ASCII grid format"""
    # Extract transform and CRS from template
    if isinstance(rastemp, str):
        with rasterio.open(rastemp) as src:
            transform = src.transform
            crs = src.crs
    else:
        transform = rastemp.transform
        crs = rastemp.crs
    
    # Write to a temporary GeoTIFF first
    temp_tif = filename.replace('.asc', '_temp.tif')
    height, width = arr_data.shape
    
    with rasterio.open(
        temp_tif,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=arr_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr_data, 1)
    
    # Now convert to ASCII using rasterio's copy function
    with rasterio.open(temp_tif) as src:
        with rasterio.open(
            filename,
            'w',
            driver='AAIGrid',
            height=height,
            width=width,
            count=1,
            dtype=arr_data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(src.read())
    
    # Remove temporary file
    os.remove(temp_tif)
    return print('gdal_writeasc successful!')


def raster_aggregate(srcfile, outfile, aggregate_level):
    """Aggregate raster by a factor"""
    with rasterio.open(srcfile) as src:
        # Calculate new dimensions
        width = int(src.width / aggregate_level)
        height = int(src.height / aggregate_level)
        
        # Create new transform
        new_transform = src.transform * rasterio.Affine.scale(aggregate_level, aggregate_level)
        
        # Create output dataset
        with rasterio.open(
            outfile,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.float32,
            crs=src.crs,
            transform=new_transform,
        ) as dst:
            # Resample data
            data = src.read(
                out_shape=(1, height, width),
                resampling=rasterio.warp.Resampling.min
            )
            dst.write(data)
    
    return 0

def flt2tif(fltfile, target_file):
    """Convert FLT file to GeoTIFF"""
    # Open the FLT file
    with rasterio.open(fltfile) as src:
        # Read data
        data = src.read(1)
        
        # Replace -999 with NaN
        data[data == -999] = np.nan
        
        # Get metadata
        meta = src.meta.copy()
        
        # If needed, update projection
        if platform == 'linux':  # on Spartan
            prj_path = '/home/yuerongz/punim0728/WHProj/gis/projection.prj'
        else:
            prj_path = 'C:/Users/mike-u/Desktop/YUERONG/TUFLOW_WilliamHovell/model/gis/projection.prj'
        
        with open(prj_path, 'r') as f:
            prj_txt = f.read()
        
        meta.update({
            'driver': 'GTiff',
            'crs': CRS.from_wkt(prj_txt)
        })
        
        # Write to new file
        with rasterio.open(target_file, 'w', **meta) as dst:
            dst.write(data, 1)
    return 0

def save_to_asc(dem_file, tensor, filename):

    # Ensure tensor is a numpy array
    if not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)
    
    # Extract transform and CRS from DEM file
    with rasterio.open(dem_file) as src:
        transform = src.transform
        crs = src.crs
    
    # Write to ASCII using the existing function
    gdal_writeasc(filename, tensor, rastemp={'transform': transform, 'crs': crs})
    
    logger.info(f"Data successfully written to {filename}")
    return 0
