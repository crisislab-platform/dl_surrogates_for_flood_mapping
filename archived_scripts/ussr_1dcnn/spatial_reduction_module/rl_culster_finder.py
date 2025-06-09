import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from modules.models.usrr_1dcnn.lib.gdal_lib import gdal_asarray, gdal_transform, rc2coords, read_shp_point
import logging

class RLClusterFinder:
    """
    Class to find clusters of representative locations using k-means algorithm
    """
    def __init__(self, output_dir, run_id, rl_file_path, sampling_distance, n_clusters=10, random_state=42, n_init=10):
        self.output_dir = output_dir
        self.run_id = run_id
        self.coordinates_file = rl_file_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.sampling_distance = sampling_distance
        
        if os.path.exists(rl_file_path):
            self.rl_locations = read_shp_point(rl_file_path)
        else:
            raise ValueError(f"RL file path is not valid: {rl_file_path}")
        
        self.output_file = os.path.join(self.output_dir, f'rl_clusters_{run_id}_{sampling_distance}.csv')
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def load_coordinates(self, coordinates_file):
        self.logger.info(f"Loading coordinates from {coordinates_file}")
        try:
            # Assuming coordinates are stored in a numpy array or similar format
            return read_shp_point(coordinates_file)
            
        except Exception as e:
            self.logger.error(f"Error loading coordinates: {e}")
            raise
            
    def cluster_locations(self, coordinates, n_clusters, random_state=42):
        self.logger.info(f"Clustering {len(coordinates)} locations into {n_clusters} clusters")
        labels = self.kmeans.fit_predict(coordinates)
        self.logger.info("Clustering completed")
        return labels

        
    def visualize_clusters(self, coordinates, labels, save_path=None):
        self.logger.info("Visualizing clusters")
        
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
            self.logger.warning(f"Cannot visualize {dim}-dimensional data directly")
            return
            
        # Save or show the visualization
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
            
    def run_clustering(self, visualize=True):
        # Load coordinates
        coordinates = self.load_coordinates(self.coordinates_file)
        
        # Convert coordinates to numpy array if it's a list
        if isinstance(coordinates, list):
            coordinates = np.array(coordinates)
        
        # Run clustering
        labels = self.cluster_locations(coordinates, self.n_clusters)
        
        # Save points and labels to csv
        output_file = self.output_file
        
        # Create array with coordinates and cluster labels
        result_data = np.column_stack((coordinates, labels))
        
        # Define header based on dimensionality
        header = "X,Y,Cluster"

        # Save results to CSV
        np.savetxt(output_file, result_data, delimiter=',', header=header, comments='')
        self.logger.info(f"Saved clustering results to {output_file}")
    
        # Visualize if requested and possible
        if visualize and hasattr(coordinates, 'shape') and len(coordinates.shape) > 1 and coordinates.shape[1] in [2, 3]:
            viz_path = os.path.join(self.output_dir, f"{self.run_id}_visualization.png")
            self.visualize_clusters(coordinates, labels, viz_path)
            
        return output_file


