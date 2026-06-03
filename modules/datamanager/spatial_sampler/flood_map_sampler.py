import numpy as np
import torch
import logging
import rasterio
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray
from modules.lib.constants import DEM_FILE, OUTPUT_DIR
from modules.lib.gdal_lib import gdal_writetiff
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import distance_transform_edt
from functools import partial

from modules.utils.run_util import check_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FloodMapSampler")

class FloodMapSampler():
    
    def __init__(self, sampling_dist, run_dir, tile_resolution,sample_flood_map, flood_map_shape):
        self.run_dir = run_dir
        self.tile_resolution = tile_resolution
        self.sampling_distance = sampling_dist
        self.sample_flood_map = sample_flood_map
        self.flood_map_shape = flood_map_shape
        self.device = check_device()
        logger.info("Device for FloodMapSampler: {}".format(self.device))
        self.dem_file = DEM_FILE
        self.number_of_tiles_per_timestep = None
        
        # To create overlapping tiles, we need to first create two channels of the original map with different starting points. One starts at (0,0) and the other starts at (cross_tile_dist, cross_tile_dist). 
        # Then we can apply the same tiling function to both channels and concatenate the results. 
        # This way we can ensure that the tiles are created with the desired overlap.
        # The number of overlapping tiles per axes (y and x) can be calculated as the sample map size  divided by the cross tile distance.
        self.num_of_nodes_per_axes = round(self.tile_resolution / self.sampling_distance)  
        self.build_reshaping_func_seq(self.num_of_nodes_per_axes)
        self.prepare_dem_template()
        
        
        logger.info("Preparation of reshaping functions completed")
        self.reconstruction_init()
        self.test_reconstruction()
        logger.info("FloodMapSampler initialized successfully")
        
    def test_reconstruction(self):
        sample_flood_map = self.process_inundation_file(self.sample_flood_map)
        sample_flood_map = torch.from_numpy(sample_flood_map).float().to(self.device)  
        sample_flood_map = sample_flood_map.unsqueeze(0).unsqueeze(0)
        flood_map_tiles = self.tiles_prep_func_gpu(sample_flood_map)
        reconstructed_map = self.reconstruct_full_map(flood_map_tiles).to(self.device)
        
        #Visualise both maps
        gdal_writetiff(sample_flood_map.squeeze(0).squeeze(0).cpu().numpy(), f"{OUTPUT_DIR}/test_reconstruction_original.tif", self.dem_file)
        gdal_writetiff(reconstructed_map.cpu().numpy(), f"{OUTPUT_DIR}/test_reconstruction_reconstructed.tif", self.dem_file)
        
        # calculate the loss between the original and reconstructed map to check if the reconstruction is working correctly. The loss should be very close to 0.
        loss = torch.nn.functional.mse_loss(reconstructed_map, sample_flood_map.squeeze(0).squeeze(0))
        logger.info(f"Reconstruction test completed with MSE loss: {loss.item()}")
        
        del sample_flood_map
        del flood_map_tiles
        del reconstructed_map
        if loss.item() > 1e-6:
            logger.warning("Reconstruction test failed with high MSE loss. Please check the tiling and reconstruction functions.")
        else:
            logger.info("Reconstruction test passed with low MSE loss.")
    
    def prepare_dem_template(self):
        # Prepare the DEM template for the tiles. This is used for the DEM conditioning in the diffusion model. 
        # The DEM template is created by tiling the original DEM to the same shape as the flood maps, and then applying the same tiling function to create the tiles.
            # Get the no data value from DEM
            
        # Print cuda memory summary before processing the DEM to help with debugging memory issues
        logger.info(f"CUDA memory summary before preparing DEM template: {torch.cuda.memory_summary(device=self.device)}")
        with rasterio.open(self.dem_file) as src:
            no_data_value = src.nodata
            dem_data = src.read(1)  # Read the first band of the DEM data
        
        dem_tensor = torch.from_numpy(dem_data).float().to(self.device)  # Load the DEM data and convert to tensor
        dem_tensor = dem_tensor.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, H, W)
        dem_template_tiles = self.tiles_prep_func_gpu(dem_tensor)
        
        
        no_data_mask = (dem_template_tiles == no_data_value)

        # Fill no_data values with nearest neighbor valid values
        dem_filled = dem_template_tiles.clone()
        for i in range(dem_filled.shape[0]):
            valid_mask = ~no_data_mask[i]  # True where data is valid
            if valid_mask.any():  # If there are any valid values
                # Use distance_transform_edt to find nearest valid pixel for each no_data pixel
                # Returns indices of nearest valid pixel
                distances, indices = distance_transform_edt(no_data_mask[i].cpu().numpy(), return_distances=True, return_indices=True)
                
                # indices is (2, H, W) where indices[0] contains row indices and indices[1] contains col indices
                nearest_rows = indices[0]
                nearest_cols = indices[1]
                
                # Get values from nearest valid pixels and assign to no_data pixels
                nearest_values = dem_filled[i, nearest_rows, nearest_cols]
                dem_filled[i, no_data_mask[i]] = nearest_values[no_data_mask[i]]
            else:  # Tile is 100% no_data - use a default (0 or mean of DEM)
                raise ValueError("Tile with 100% no-data values found. Please check the DEM data and tiling function.")
                
        # Normalize the DEM template tiles to the range [-1,1] for better training stability. 
        # Use minmax normalisation
        dem = dem_filled.squeeze(1)  # Remove channel dimension for normalization
        # Normalize to log1p to match the distribution of the flood maps.
        #make the minimum value of the DEM 0 by subtracting the minimum value from all values. This is important for the log1p transformation to avoid issues with negative values.
        dem = dem - dem.min()
        # Now do min max normalization to [0,1]
        scaler = MinMaxScaler()
        dem_shape = dem.shape
        dem = scaler.fit_transform(dem.cpu().numpy().reshape(-1, dem_shape[-1])).reshape(dem_shape)
        dem = torch.from_numpy(dem).float().to(self.device)
        self.dem_template_tiles = dem.detach().cpu()

        del dem_tensor
        del dem_filled
        del dem
        del no_data_mask
        # Check if there are still any no data values in the DEM template tiles after filling. If there are, raise an error.
        if (self.dem_template_tiles == no_data_value).any():
            logger.error("There are still no data values in the DEM template tiles after filling. Please check the DEM data and the tiling function.")
            raise ValueError("No data values found in DEM template tiles after filling.")
        
        logger.info("DEM tiles prepared successfully")
        
    def calculate_batch_size_and_number_of_tiles(self, indices_per_timestep=10):
        # Calculate the number of tiles per time step based on the sample resolution and sampling distance, and then calculate the batch size based on the desired number of batches per time step.
        num_of_tiles = self.dem_template_tiles.shape[0]
        self.number_of_tiles_per_timestep = num_of_tiles

        if num_of_tiles % indices_per_timestep != 0:
            logger.warning(f"The total number of tiles per time step ({num_of_tiles}) is not perfectly divisible by the desired number of batches per time step ({indices_per_timestep}).")
            raise ValueError("Please adjust the sampling distance or the desired number of batches per time step to ensure that the total number of tiles is divisible by the number of batches.")
        tiles_per_index = num_of_tiles // indices_per_timestep
        logger.info(f"Total number of tiles per time step: {num_of_tiles}, Tiles per batch index: {tiles_per_index}")
        return  tiles_per_index, num_of_tiles
        

    def build_reshaping_func_seq(self, num_of_sub_maps_per_axes):
        # Function sequence to create reshaped maps(tiles) used for training. 
        # Multiple functions are created for creating different tiles from channels with different starting points to achieve overlapping tiles.
        self.reshaping_func_seq = []
        for r_i in range(num_of_sub_maps_per_axes):
            self.reshaping_func_seq.append(self.build_tiling_func(r_i * self.sampling_distance, r_i * self.sampling_distance))
            
    def tiles_prep_func_gpu(self, x):
            with torch.no_grad():
                tiles_maps =  [tiling_func(x) for tiling_func in self.reshaping_func_seq]
                tiles_maps = torch.cat(tiles_maps, dim=0)
                return tiles_maps
    
    def get_padding_height_width(self):
        # Calculate padding needed for height and width to be divisible by map_size
        orig_height, orig_width = self.flood_map_shape
        pad_height = (self.tile_resolution - (orig_height % self.tile_resolution)) % self.tile_resolution
        pad_width = (self.tile_resolution - (orig_width % self.tile_resolution)) % self.tile_resolution
        return pad_height, pad_width, orig_height, orig_width
    
    def build_tiling_func(self, cross_tile_size_r, cross_tile_size_c, training_mode=True):
        pad_height, pad_width, orig_height, orig_width = self.get_padding_height_width()
        sample_flood_map = self.process_inundation_file(self.sample_flood_map)
        sample_flood_map = torch.from_numpy(sample_flood_map)
        # Load the sample flood map and convert to tensor
        sample_flood_map = sample_flood_map.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, H, W)

        # Create padding function using partial instead of nested function
        padding_func = partial(self._padding_func, pad_height=pad_height, pad_width=pad_width,
                               cross_tile_size_r=cross_tile_size_r, cross_tile_size_c=cross_tile_size_c)
        
    
        new_shape = padding_func(sample_flood_map).shape
        ax0_n = new_shape[2] // self.tile_resolution
        ax1_n = new_shape[3] // self.tile_resolution
        logger.info(f"New shape after padding: {new_shape}, maps: {ax0_n}x{ax1_n}")
        
        # Create a reshaping function to get the maps(tiles) for this specific starting point (cross_tile_size_r, cross_tile_size_c)
        reshaping_func = partial(self.reshaping_func_def, padding_func, ax0_n=ax0_n, ax1_n=ax1_n)
        
        # Filter arrays with no-data dem tiles.
        with rasterio.open(self.dem_file) as src:
            no_data_value = src.nodata
            dem_data = src.read(1)  # Read the first band of the DEM data
            
        dem_data = torch.from_numpy(dem_data).float().unsqueeze(0).unsqueeze(0)  # Load the DEM data and convert to tensor
        dem_tiles = reshaping_func(dem_data)
       
        # Only filter out tiles with 100% no-data values (completely empty tiles)
        # Tiles with sparse data (70-99% no-data) still have valid pixels that contribute to reconstruction
        no_data_mask = (dem_tiles == no_data_value)
        no_data_ratio = no_data_mask.sum(axis=(1, 2)) / (self.tile_resolution ** 2)
        dem_tile_filter_arr = no_data_ratio < 1.0  #Keep tiles with any valid data

        if training_mode:
            tiling_func = partial(self._apply_tile_filter, reshaping_func, dem_tile_filter_arr)
            logger.info(f"Created tiling function successfully")
            return tiling_func
        else:
            return reshaping_func, dem_tile_filter_arr, ax0_n, ax1_n, (cross_tile_size_r, cross_tile_size_c)

    def reshaping_func_def(self, slicing_func, x, ax0_n=None, ax1_n=None):
        # First reshape the map to the desired size (a multiple of the map size)
        new_map_x = slicing_func(x)[:, :, :self.tile_resolution * ax0_n, :self.tile_resolution * ax1_n]
        del x  # Free memory after slicing
        new_map_x = new_map_x.to(self.device)
        # Shape of the new map is (1, 1, sample_resolution * ax0_n, sample_resolution * ax1_n). Remove batch and channel dims.
        new_map_x = new_map_x.squeeze(0).squeeze(0)  # (1, 1, H, W) -> (H, W)
        new_map_x = new_map_x.reshape((-1, ax1_n, self.tile_resolution)) 
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((ax1_n, -1, self.tile_resolution, self.tile_resolution))
        new_map_x = new_map_x.swapaxes(0, 1)
        new_map_x = new_map_x.reshape((-1, self.tile_resolution, self.tile_resolution))
        return new_map_x.detach().cpu()
    
    def _apply_tile_filter(self, reshaping_func, dem_tile_filter_arr, x):
        """Apply tile filter - pickle-compatible helper for tiling_func"""
        return reshaping_func(x)[dem_tile_filter_arr, :, :]
    
    def _back_transform(self, ax1_n, x):
        """Back transformation helper - pickle-compatible alternative to lambda"""
        x = x.to(self.device)
        x = x.reshape(-1, ax1_n, self.tile_resolution, self.tile_resolution)
        x = x.swapaxes(0, 1)
        x = x.reshape(ax1_n, -1, self.tile_resolution)
        x = x.swapaxes(0, 1)
        x = x.reshape(-1, ax1_n * self.tile_resolution)
        return x.detach().cpu()
    
    def _padding_func(self, x, pad_height=None, pad_width=None, cross_tile_size_r=None, cross_tile_size_c=None):
        """Padding helper - pickle-compatible alternative to nested function"""
        # x is expected to be of shape (1, 1, H, W)
        # Use constant padding (0) instead of reflect to avoid boundary artifacts that misalign during reconstruction
        x = x.to(self.device)
        padded = torch.nn.functional.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0)
        sliced = padded[:, :, cross_tile_size_r:, cross_tile_size_c:]  # Slice to create the tiled version with desired offset
        del padded  # Free memory after slicing
        del x  # Free memory after padding
        return sliced.detach().cpu()

    def reconstruction_init(self):
        logger.info("Initializing reconstruction")
        self.output_maps_list = []
        self.return_template_list = []
        self.tile_filter_list = []
        self.layers_seperation_idxs = []
        self.map_origins = []
        self.layer_sizes = []
        self.back_transformation_functions = []
        start_idx = 0  
        
        with torch.no_grad():
            for r_i in range(self.num_of_nodes_per_axes):
                c_i = r_i
                reshaping_func, filter_arr, ax0_n, ax1_n, map_origin = \
                    self.build_tiling_func(r_i * self.sampling_distance, 
                                           c_i * self.sampling_distance, training_mode=False)

                self.layers_seperation_idxs.append((start_idx, start_idx + torch.sum(filter_arr).item()))
                start_idx = self.layers_seperation_idxs[-1][1]
                pad_height, pad_width, original_height, original_width = self.get_padding_height_width()
                output_map = torch.zeros((original_height + pad_height, original_width + pad_width), dtype=torch.float32)
                self.output_maps_list.append(output_map)
                output_map = output_map.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, H, W)
                self.return_template_list.append(reshaping_func(output_map))
                self.tile_filter_list.append(filter_arr.bool())
                # Store origin in the UNPADDED coordinate space for proper reconstruction placement
                # This ensures that overlapping tiles align correctly regardless of padding artifacts
                self.map_origins.append(map_origin)
                self.layer_sizes.append((ax0_n*self.tile_resolution, ax1_n*self.tile_resolution))
                self.back_transformation_functions.append(self.build_back_func_lambda(ax1_n))
                
            # Check if the reconstruction output covers the entire map
            temp_tensor = torch.ones(self.dem_template_tiles.shape)
            self.final_map = self.reconstruct_full_map_return_and_sum(temp_tensor).cpu()
            # To avoid division by zero during reconstruction. This means that the areas not covered by any tile will be left unchanged during reconstruction, which is the desired behavior.
            self.final_map[self.final_map == 0] = 1.0
            del temp_tensor
            logger.info("Reconstruction initialization completed successfully")
        return 0

    def reconstruct_full_map_return_and_sum(self, x): 
        for lyr_i in range(len(self.layers_seperation_idxs)):
            start_idx, end_idx = self.layers_seperation_idxs[lyr_i]
            self.return_template_list[lyr_i][self.tile_filter_list[lyr_i]] = x[start_idx:end_idx].squeeze(1)
            row_size, col_size = self.layer_sizes[lyr_i]
            row_origin, col_origin = self.map_origins[lyr_i]
            output_map_for_layer = self.back_transformation_functions[lyr_i](self.return_template_list[lyr_i]) 
            self.output_maps_list[lyr_i][row_origin:row_origin + row_size, col_origin:col_origin + col_size] = \
                output_map_for_layer
        output_sum = torch.sum(torch.stack(self.output_maps_list), dim=0)
        return output_sum
    
    def reconstruct_full_map(self, x):
        with torch.no_grad():
            output_final = torch.div(self.reconstruct_full_map_return_and_sum(x), self.final_map)
            pad_height, pad_width, original_height, original_width = self.get_padding_height_width()
            output_final = output_final[:original_height, :original_width]
        return output_final
    
    def build_back_func_lambda(self, ax1_n):
        return partial(self._back_transform, ax1_n)
    
    def process_inundation_file(self, file_path):
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                nodata_value = src.nodata
                data = np.where(np.isnan(data), 0, data)
                data = np.where(data == nodata_value, 0, data)
                
                return data
        except Exception as e:
            logger.error(f"Error processing inundation file {file_path}: {e}")
            return None  
        
        
    def load_dem(self, normalize=True):
        with rasterio.open(DEM_FILE) as src:
            no_data = src.nodata
            dem_data = src.read(1)
            max_dem_value = np.nanmax(dem_data)
            dem_data = np.where(dem_data == no_data, max_dem_value, dem_data)
            dem_data = np.where(np.isnan(dem_data), max_dem_value, dem_data)
            dem_tensor = torch.from_numpy(dem_data).float()
            if normalize:
                dem_tensor = (dem_tensor - dem_tensor.min()) / (dem_tensor.max() - dem_tensor.min())
            return dem_tensor