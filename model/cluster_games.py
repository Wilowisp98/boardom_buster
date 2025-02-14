import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
from typing import List, Dict, Any

RELEVANT_COLUMNS: List[str] = [
    "AGE_GROUP",
    "GAME_CAT",
    "LANGUAGE_DEPENDENCY",
    "GAME_DURATION"
]

def cluster_all_games(df: pl.DataFrame, min_cluster_size: int = 10) -> List[Dict[str, Any]]:
    """
    Clusters the entire dataset using HDBSCAN.

    Args:
        df (pl.DataFrame): The DataFrame containing all board games.
        min_cluster_size (int): Minimum number of points per cluster.

    Returns:
        List[Dict[str, Any]]: A list of clusters, each containing members, a centroid (as an array), and cluster size.
    """
    # Select relevant feature columns
    relevant_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in RELEVANT_COLUMNS)]
    feature_df = df.select(relevant_columns)
    
    # Convert Polars DataFrame to NumPy array for clustering
    feature_matrix = feature_df.to_numpy()
    
    # Normalize feature data
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    cluster_labels = clusterer.fit_predict(feature_matrix_scaled)
    
    # Add cluster labels to the DataFrame
    df = df.with_columns(pl.Series("cluster_label", cluster_labels))
    
    # Group by clusters and compute centroids
    clusters = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Ignore noise points
            continue
        
        cluster_games = df.filter(pl.col("cluster_label") == cluster_id)
        cluster_size = cluster_games.height
        
        # Compute the centroid manually as the mean of all points in the cluster
        cluster_points = feature_matrix_scaled[cluster_labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)  # Centroid as an array
        
        clusters.append({
            "cluster_id": cluster_id,
            "games": cluster_games["game_name"].to_list(),
            "size": cluster_size,
            "centroid": centroid.tolist()  # Store centroid as a list for JSON serialization
        })
        
    return clusters