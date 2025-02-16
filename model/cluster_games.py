import polars as pl
import numpy as np
import hdbscan
from typing import List, Dict, Any
from sklearn.manifold import TSNE

RELEVANT_COLUMNS: List[str] = [
    "AGE_GROUP",
    "GAME_CAT",
    "LANGUAGE_DEPENDENCY",
    "GAME_DURATION"
]

def get_feature_columns(df: pl.DataFrame) -> List[str]:
    """
    Gets all feature columns based on prefixes.
    """
    feature_columns = []
    for prefix in RELEVANT_COLUMNS:
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        feature_columns.extend(matching_cols)
    return feature_columns

def cluster_all_games(df: pl.DataFrame, min_cluster_size: int = 10) -> List[Dict[str, Any]]:
    """
    Clusters the entire dataset using HDBSCAN with Jaccard distance.

    Args:
        df (pl.DataFrame): The DataFrame containing all board games.
        min_cluster_size (int): Minimum number of points per cluster.

    Returns:
        List[Dict[str, Any]]: A list of clusters, each containing members, a centroid, and cluster size.
    """
    # Select relevant feature columns
    feature_columns = get_feature_columns(df)
    feature_df = df.select(feature_columns)
    
    # Convert Polars DataFrame to NumPy array for clustering
    feature_matrix = feature_df.to_numpy()
    
    # Reduce dimensionality using t-SNE (optional but recommended for high-dimensional binary data)
    tsne = TSNE(n_components=2, random_state=42)
    feature_matrix_reduced = tsne.fit_transform(feature_matrix)
    
    # Apply HDBSCAN clustering with Jaccard distance
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='jaccard')
    cluster_labels = clusterer.fit_predict(feature_matrix_reduced)
    
    # Add cluster labels to the DataFrame
    df = df.with_columns(pl.Series("cluster_label", cluster_labels))
    
    # Group by clusters and compute centroids
    clusters = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
        
        cluster_games = df.filter(pl.col("cluster_label") == cluster_id)
        cluster_size = cluster_games.height
        
        # Compute the centroid as the mean of all points in the cluster
        cluster_points = feature_matrix[cluster_labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)  # Centroid as an array of probabilities
        
        clusters.append({
            "cluster_id": cluster_id,
            "games": cluster_games["game_name"].to_list(),
            "size": cluster_size,
            "centroid": centroid.tolist()
        })
        
    return clusters