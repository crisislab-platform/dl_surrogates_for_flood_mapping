import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from modules.models.usrr_1dcnn.lib.gdal_lib import read_shp_point, gdal_asarray
from modules.models.usrr_1dcnn.lib.base_functions import save_pts_to_shp
import logging
from modules.lib.constants import PLOTS_OUTPUT_DIR
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLClusterFinder:
    """
    Class to find clusters of representative locations using k-means algorithm
    """
    def __init__(self, output_dir, run_id, rl_file_path, sampling_distance, n_clusters=10, random_state=42, n_init=10, raster_temp=None):
        self.output_dir = os.path.join(output_dir, "clusters")
        self.run_id = run_id
        self.coordinates_file = rl_file_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.sampling_distance = sampling_distance
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.raster_temp = raster_temp
        os.makedirs(self.output_dir, exist_ok=True)
            
    def cluster_locations(self, coordinates, n_clusters, random_state=42):
        logger.info(f"Clustering {len(coordinates)} locations into {n_clusters} clusters")
        labels = self.kmeans.fit_predict(coordinates)
        logger.info("Clustering completed")
        return labels

            
    def run_clustering(self, visualize=True):
        coords_df = pd.read_csv(self.coordinates_file)
        coordinates = coords_df[['x', 'y']].values
        coordinates = np.array(coordinates)
        labels = self.cluster_locations(coordinates, self.n_clusters)
    
        coords_df['cluster'] = labels
        
        # Save all clusters to a single CSV file
        out_file = os.path.join(self.output_dir, f'clusters_ss_{self.sampling_distance}_{self.n_clusters}.csv')
        coords_df.to_csv(out_file, index=False)
        logger.info(f"Saved all clusters to a single CSV file: {out_file}")
        
        # Visualize if requested and possible
        if visualize and hasattr(coordinates, 'shape') and len(coordinates.shape) > 1 and coordinates.shape[1] in [2, 3]:
            viz_path = os.path.join(PLOTS_OUTPUT_DIR, f'clusters_{self.sampling_distance}.png')
            # self.visualize_clusters(coordinates, labels, viz_path)
            
    def run_clustering_new(self, visualize=True):
        rl_arr = gdal_asarray(self.coordinates_file)
        rows, cols = np.where(rl_arr == 1)
        
        # Create a coordinate array for clustering
        coordinates = np.column_stack((rows, cols)) 
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        coordinates_scaled = scaler.fit_transform(coordinates)  
        logger.info(f"Found {len(coordinates)} representative locations for clustering")
        
        # Apply K-means clustering
        labels = self.cluster_locations(coordinates_scaled, self.n_clusters, self.random_state)
  
        # Create a dataframe with the results
        result_df = pd.DataFrame({
            'row': rows,
            'col': cols,
            'cluster': labels
        })
        
        # Save results to CSV
        out_file = os.path.join(self.output_dir, f'clusters_ss_{self.sampling_distance}_{self.n_clusters}.csv')
        result_df.to_csv(out_file, index=False)
        logger.info(f"Saved cluster assignments to: {out_file}")
        
        self.visualise_clusters(rows, cols, labels, rl_arr)
        
    def visualise_clusters(self, rows, cols, labels, rl_arr):
        # Load DEM as background instead of empty raster
        dem_raster = gdal_asarray(self.raster_temp)
        
        # Create figure with DEM background
        plt.figure(figsize=(12, 10))
        
        # Plot DEM with terrain colormap
        dem_plot = plt.imshow(dem_raster, cmap='terrain', alpha=0.8)
        dem_cbar = plt.colorbar(dem_plot, shrink=0.7, pad=0.01, label='Elevation')
        
        # Create a scatter plot for clusters with different colors
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('viridis', len(unique_labels))
        
        # Plot each cluster with a different color
        scatter = plt.scatter(cols, rows, c=labels, cmap=cmap, s=30, 
                             edgecolor='black', linewidth=0.5)
        
        # Calculate cluster sizes once before the loop
        cluster_sizes = {label_id: np.sum(labels == label_id) for label_id in unique_labels}
        
        # Select top 5 largest clusters to annotate
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        clusters_to_annotate = [c[0] for c in sorted_clusters[:5]]
        
        # Add specific clusters to annotate if they exist
        special_clusters = [0]
        if 199 in unique_labels:
            special_clusters.append(199)
            
        # Combine with top clusters, avoiding duplicates
        for cluster in special_clusters:
            if cluster in unique_labels and cluster not in clusters_to_annotate:
                clusters_to_annotate.append(cluster)
        
        # Add annotations for each cluster
        for label_id in unique_labels:
            # Find points in this cluster
            mask = labels == label_id
            if np.sum(mask) > 0:
                # Calculate the centroid of this cluster
                center_row = int(np.mean(rows[mask]))
                center_col = int(np.mean(cols[mask]))
                
                if label_id in clusters_to_annotate:
                    # Special annotation format for clusters 0 and 199
                    if label_id in special_clusters:
                        annotation_text = f'C{label_id+1}\n#{cluster_sizes[label_id]} cells'
                    else:
                        annotation_text = f'C{label_id+1}'
                        
                    # Add a text annotation
                    plt.annotate(
                        annotation_text,
                        (center_col, center_row),
                        fontsize=8,  # Smaller font
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6, ec='none'),  # Smaller padding
                        ha='center',
                        va='center'
                    )
        
        plt.title(f'Representative Location Clusters (n={self.n_clusters}) on DEM')
        plt.xlabel('X (Column)')
        plt.ylabel('Y (Row)')
        
        # Save the visualization
        raster_viz_path = os.path.join(PLOTS_OUTPUT_DIR, f'cluster_raster_{self.sampling_distance}.png')
        plt.savefig(raster_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved cluster visualization on DEM to: {raster_viz_path}")







