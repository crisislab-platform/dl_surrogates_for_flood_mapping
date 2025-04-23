import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from modules.models.usrr_1dcnn.spatial_reduction_module.gdal_lib import read_shp_point
from modules.models.usrr_1dcnn.spatial_reduction_module.base_functions import save_pts_to_shp
import logging
from modules.lib.constants import GRAPH_OUTPUT_DIR
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLClusterFinder:
    """
    Class to find clusters of representative locations using k-means algorithm
    """
    def __init__(self, output_dir, run_id, rl_file_path, sampling_distance, n_clusters=10, random_state=42, n_init=10):
        self.output_dir = os.path.join(output_dir, "clusters")
        self.run_id = run_id
        self.coordinates_file = rl_file_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.sampling_distance = sampling_distance
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        os.makedirs(self.output_dir, exist_ok=True)
            
    def cluster_locations(self, coordinates, n_clusters, random_state=42):
        logger.info(f"Clustering {len(coordinates)} locations into {n_clusters} clusters")
        labels = self.kmeans.fit_predict(coordinates)
        logger.info("Clustering completed")
        return labels

        
    def visualize_clusters(self, coordinates, labels, save_path=None):
        logger.info("Visualizing clusters")
        
        # Determine dimensionality of the data
        dim = coordinates.shape[1]
        if dim == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
            plt.title('Location Clusters')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.colorbar(label='Cluster')
            plt.grid(True, linestyle='--', alpha=0.7)
            
        elif dim == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
                               c=labels, cmap='viridis', marker='o', alpha=0.6)
            plt.title('Location Clusters')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            plt.colorbar(scatter, label='Cluster')
            
        else:
            logger.warning(f"Cannot visualize {dim}-dimensional data directly")
            return
            
        # Save or show the visualization
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
            
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
            viz_path = os.path.join(GRAPH_OUTPUT_DIR, f'clusters_{self.sampling_distance}.png')
            self.visualize_clusters(coordinates, labels, viz_path)




