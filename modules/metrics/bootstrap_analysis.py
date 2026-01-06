import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import logger
from modules.lib.constants import OUTPUT_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BootstrapAnalysis")

def bootstrap_function(model_name, residual_maps, reference_maps, num_samples=100):
    #overall RMSE
    overall_rmse = torch.sqrt(torch.mean(residual_maps**2))
    logger.info(f"Overall RMSE: {overall_rmse.item()}")
    
    #reshape residual maps back to (timesteps, rows, cols)
    residual_maps = residual_maps.view(residual_maps.shape[0], 611, 951)
    
    # Use GPU for faster processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_maps = residual_maps.to(device).float()
    
    # Optimized block bootstrap
    bootstrap_rmses = np.zeros(num_samples)
    wet_rmse_values = np.zeros(num_samples)
    nse_values = np.zeros(num_samples)
    mae_values = np.zeros(num_samples)
    
    # Define parameters
    temp_block_size = 24
    spatial_block_size = (50, 50)
    
    wet_cells_mask = (reference_maps > 0.3).float().to(device)
    
    for i in range(num_samples):
        torch.cuda.empty_cache()
        logger.info(f"Sampling bootstrap iteration {i+1}/{num_samples}")
        
        # 1. Temporal block bootstrap (much faster approach)
        # Generate random indices for temporal blocks all at once
        num_blocks_needed = (266 + temp_block_size - 1) // temp_block_size
        max_start_idx = 266 - temp_block_size + 1
        
        # Generate random start positions for temporal blocks
        block_starts = torch.randint(0, max_start_idx, (num_blocks_needed,), device=device)
        
        # Create index tensor for all positions we need
        indices = []
        for start in block_starts:
            indices.append(torch.arange(start, start + temp_block_size, device=device))
        
        # Concatenate and trim
        time_indices = torch.cat(indices)[:266]
        
        # Use advanced indexing to select all blocks at once (much faster)
        bootstrap_sample = residual_maps[time_indices]
        
        # 2. Spatial block bootstrap using 2D convolution for efficiency
        # We'll use the random sampling of locations but in a batched operation
        bs_height, bs_width = bootstrap_sample.shape[1], bootstrap_sample.shape[2]
        
        # Create an output tensor initialized with zeros on GPU
        spatially_bootstrapped = torch.zeros_like(bootstrap_sample, device=device)
        
        # Use torch.nn.functional.grid_sample for extremely efficient resampling of ALL timesteps at once
        import torch.nn.functional as F
        
        # Convert bootstrap sample to batch format (N,C,H,W) where N=timesteps, C=1
        resampled_data = bootstrap_sample.unsqueeze(1)  # Shape: [timesteps, 1, height, width]
        
        # Create sampling grid coordinates
        # For block bootstrapping, we need to generate a grid that samples blocks rather than individual pixels
        # We'll create a grid that maps from target positions to randomly selected source blocks
        
        # Number of blocks in each dimension
        n_blocks_h = bs_height // spatial_block_size[0]
        n_blocks_w = bs_width // spatial_block_size[1]
        
        # Create random mapping for blocks
        # For each target block position, select a random source block
        block_map_h = torch.randint(0, n_blocks_h, (n_blocks_h,), device=device)
        block_map_w = torch.randint(0, n_blocks_w, (n_blocks_w,), device=device)
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(bs_height, device=device).float() / (bs_height - 1) * 2 - 1,
            torch.arange(bs_width, device=device).float() / (bs_width - 1) * 2 - 1,
            indexing='ij'
        )
        
        # Convert grid coordinates to block indices
        block_idx_y = torch.floor(((grid_y + 1) / 2) * n_blocks_h).long().clamp(0, n_blocks_h-1)
        block_idx_x = torch.floor(((grid_x + 1) / 2) * n_blocks_w).long().clamp(0, n_blocks_w-1)
        
        # Map block indices through our random mapping
        mapped_block_idx_y = block_map_h[block_idx_y]
        mapped_block_idx_x = block_map_w[block_idx_x]
        
        # Convert mapped block indices back to normalized coordinates for grid_sample
        # Add random offsets within the block for smoother borders
        block_size_norm_h = 1.0 / n_blocks_h
        block_size_norm_w = 1.0 / n_blocks_w
        
        # Calculate source coordinates
        src_y = (mapped_block_idx_y.float() + (block_idx_y.float() % 1.0)) * block_size_norm_h * 2 - 1
        src_x = (mapped_block_idx_x.float() + (block_idx_x.float() % 1.0)) * block_size_norm_w * 2 - 1
        
        # Create the sampling grid for grid_sample
        sampling_grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0)  # Shape: [1, H, W, 2]
        
        # Perform the resampling in one efficient operation for all timesteps
        # This is vastly more efficient than the loop-based approach
        spatially_bootstrapped = F.grid_sample(
            resampled_data, 
            sampling_grid.repeat(resampled_data.shape[0], 1, 1, 1),
            mode='nearest',  # Use 'nearest' for block resampling
            align_corners=True
        ).squeeze(1)  # Remove the channel dimension
        
        
        # Calculate RMSE efficiently with a single operation, with better error handling
        logger.info(f"Calculating RMSE for bootstrap sample {i}")
        logger.info(spatially_bootstrapped.shape)
        
        # Clear CUDA cache to avoid memory issues
        torch.cuda.empty_cache()
        
        try:
            mse = torch.mean(spatially_bootstrapped**2)
            rmse = torch.sqrt(mse)
            
            # Calculate other metrics
            mae = torch.mean(torch.abs(spatially_bootstrapped))
            
            # Get reference data for this bootstrap sample
            bootstrapped_reference = reference_maps.view(reference_maps.shape[0], 611, 951)[time_indices]
            
            # Calculate masked RMSE (only for wet cells)
            wet_rmse, _ = mRMSE(spatially_bootstrapped.view(266, -1), 
                             bootstrapped_reference.view(266, -1))
            
            # Calculate NSE
            nse_value = NSE(spatially_bootstrapped.view(266, -1), 
                           bootstrapped_reference.view(266, -1))
            
            # Get standard RMSE value
            rmse_value = rmse.detach().cpu().item()
            
            if rmse_value is None or np.isnan(rmse_value):
                logger.warning(f"Bootstrap RMSE for sample {i} is NaN, skipping this sample.")
                continue
            
            bootstrap_rmses[i] = rmse_value
            wet_rmse_values[i] = wet_rmse
            nse_values[i] = nse_value
            mae_values[i] = mae.detach().cpu().item()
            
            logger.info(f"Bootstrap RMSE for sample {i}: {rmse_value:.4f}, Wet RMSE: {wet_rmse:.4f}, NSE: {nse_value:.4f}")
        except Exception as e:
            logger.warning(f"Error calculating RMSE for bootstrap sample {i}: {e}")
            continue
        
    # Calculate the 95% confidence interval
    bootstrap_rmses_tensor = torch.tensor(bootstrap_rmses)
    lower_ci = np.percentile(bootstrap_rmses, 2.5)
    upper_ci = np.percentile(bootstrap_rmses, 97.5)
    
    logger.info(f"Bootstrap RMSE 95% Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
    logger.info(f"Bootstrap RMSE Mean: {bootstrap_rmses_tensor.mean():.4f}")
    logger.info(f"Bootstrap RMSE Standard Deviation: {bootstrap_rmses_tensor.std():.4f}")
    
    # calculate mean wet RMSE, NSE, MAE and 95% CI
    mean_wet_rmse = np.mean(wet_rmse_values)
    mean_nse = np.mean(nse_values)
    mean_mae = np.mean(mae_values)
    lower_ci_wet_rmse = np.percentile(wet_rmse_values, 2.5)
    upper_ci_wet_rmse = np.percentile(wet_rmse_values, 97.5)
    lower_ci_nse = np.percentile(nse_values, 2.5)
    upper_ci_nse = np.percentile(nse_values, 97.5)
    lower_ci_mae = np.percentile(mae_values, 2.5)
    upper_ci_mae = np.percentile(mae_values, 97.5)
    
    
    logger.info(f"Wet RMSE Mean: {mean_wet_rmse:.4f}, 95% CI: [{lower_ci_wet_rmse:.4f}, {upper_ci_wet_rmse:.4f}]")
    logger.info(f"NSE Mean: {mean_nse:.4f}, 95% CI: [{lower_ci_nse:.4f}, {upper_ci_nse:.4f}]")  
    logger.info(f"MAE Mean: {mean_mae:.4f}, 95% CI: [{lower_ci_mae:.4f}, {upper_ci_mae:.4f}]")
    
    
    # Create an enhanced histogram plot
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE curve
    counts, bins, patches = plt.hist(bootstrap_rmses, bins=25, density=True, alpha=0.6, 
                                    color='steelblue', edgecolor='black', label='Bootstrap Samples')
    
    # Add kernel density estimate
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(bootstrap_rmses)
    x = np.linspace(min(bootstrap_rmses), max(bootstrap_rmses), 1000)
    plt.plot(x, kde(x), 'r-', linewidth=2, label='Kernel Density Estimate')
    
    # Add vertical lines for important statistics
    plt.axvline(overall_rmse.item(), color='green', linestyle='dashed', 
                linewidth=2, label=f'Overall RMSE: {overall_rmse.item():.4f}')
    plt.axvline(lower_ci, color='red', linestyle='dotted', 
                linewidth=2, label=f'95% CI Lower: {lower_ci:.4f}')
    plt.axvline(upper_ci, color='red', linestyle='dotted', 
                linewidth=2, label=f'95% CI Upper: {upper_ci:.4f}')
    
    # Add grid and improve aesthetics
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Bootstrap RMSE', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Bootstrap RMSE Distribution for {model_name} (n={num_samples})', fontsize=16)
    plt.legend(fontsize=12)
    
    # Add statistics annotation in the plot
    stats_text = f"Mean: {bootstrap_rmses_tensor.mean():.4f}\n"
    stats_text += f"Std Dev: {bootstrap_rmses_tensor.std():.4f}\n"
    stats_text += f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]"
    
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                fontsize=12, va='top')
    
    # Ensure output directory exists
    os.makedirs(os.path.join(OUTPUT_DIR, 'quality_metrics'), exist_ok=True)
    
    # Save the figure with higher DPI for better quality
    outfile = os.path.join(OUTPUT_DIR, 'quality_metrics', f'bootstrap_rmse_distribution_{model_name}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved enhanced bootstrap RMSE distribution plot at {outfile}")
    
    out_stat_file = os.path.join(OUTPUT_DIR, 'quality_metrics', f'bootstrap_rmse_stats.csv')
    if os.path.exists(out_stat_file):
        stats_df = pd.read_csv(out_stat_file)
    
    else:
        stats_df = pd.DataFrame(columns=['model', 'mean_rmse', 'std_rmse', 'ci_lower', 'ci_upper',
                                        'overall_rmse', 'mean_wet_rmse', 'ci_lower_wet_rmse', 'ci_upper_wet_rmse',
                                        'mean_nse', 'ci_lower_nse', 'ci_upper_nse',
                                        'mean_mae', 'ci_lower_mae', 'ci_upper_mae',
                                        'num_samples', 'block_size_time', 'block_size_space']) 
    new_row = {
        'model': model_name,
        'mean_rmse': bootstrap_rmses_tensor.mean().item(),
        'std_rmse': bootstrap_rmses_tensor.std().item(),
        'ci_lower': lower_ci,
        'ci_upper': upper_ci, 
        'overall_rmse': overall_rmse.item(),
        'mean_wet_rmse': mean_wet_rmse,
        'ci_lower_wet_rmse': lower_ci_wet_rmse,
        'ci_upper_wet_rmse': upper_ci_wet_rmse,
        'mean_nse': mean_nse,
        'ci_lower_nse': lower_ci_nse,
        'ci_upper_nse': upper_ci_nse,
        'mean_mae': mean_mae,
        'ci_lower_mae': lower_ci_mae,  
        'ci_upper_mae': upper_ci_mae,
        'num_samples': num_samples, 
        'block_size_time': 24,
        'block_size_space': '50x50'
    }
    
    stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)                
    stats_df.to_csv(out_stat_file, index=False)
    logger.info(f"Saved bootstrap RMSE statistics at {out_stat_file}")
    

def mRMSE(residual_maps, reference_maps):
    """
    Calculate masked RMSE for wet cells only.
    Only considers cells that are wet (>threshold) in the reference map.
    
    Args:
        residual_maps: Prediction - Reference (residuals)
        reference_maps_tensor_reshaped: Reference/ground truth values
        
    Returns:
        Tuple of (overall_mrmse, mrmse_per_timestep, mrmse_per_cell)
    """
    threshold = 0.3
    
    # Create mask for wet cells in reference (ground truth)
    wet_mask = (reference_maps > threshold).float()
    
    # Squared residuals only for wet cells
    squared_residuals = (residual_maps ** 2) * wet_mask
    
    # mRMSE per timestep
    num_wet_per_timestep = wet_mask.sum(dim=1)
    
    # Avoid division by zero
    num_wet_per_timestep = torch.where(num_wet_per_timestep > 0, num_wet_per_timestep, torch.ones_like(num_wet_per_timestep))
    mse_per_timestep = squared_residuals.sum(dim=1) / num_wet_per_timestep
    mrmse_per_timestep = torch.sqrt(mse_per_timestep)
        
    #Overall mRMSE
    mrmse = torch.mean(mrmse_per_timestep).item()
    return mrmse, mrmse_per_timestep

def NSE(residual_maps, reference_maps):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) from residual maps and reference maps
    
    Args:
        residual_maps: Tensor of residuals (pred - ref) with shape [timesteps, cells]
        reference_maps: Tensor of reference values with shape [timesteps, cells]
    
    Returns:
        NSE value as a scalar
    """
    # Reshape to [timesteps, cells] if needed
    if len(residual_maps.shape) > 2:
        residual_maps = residual_maps.view(residual_maps.shape[0], -1)
    if len(reference_maps.shape) > 2:
        reference_maps = reference_maps.view(reference_maps.shape[0], -1)
        
    # Calculate mean of observed (reference) values
    observed_mean = torch.mean(reference_maps)
    
    # Reconstruct predicted values from residuals and reference
    predicted = reference_maps + residual_maps  # Since residual = predicted - reference
    
    # Calculate NSE components
    numerator = torch.sum((reference_maps - predicted) ** 2)
    denominator = torch.sum((reference_maps - observed_mean) ** 2)
    
    # Handle edge cases
    if denominator.item() == 0.0:
        if numerator.item() == 0.0:
            return 1.0
        else:
            logger.warning("Denominator is zero in NSE calculation, returning NSE as 0")
            return 0.0
            
    # Calculate NSE
    nse = 1.0 - (numerator / denominator)
    return nse.item()