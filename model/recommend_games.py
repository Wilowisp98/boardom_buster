import polars as pl
import numpy as np
from typing import List, Dict, Any

def recommend_games(clusters: List[Dict[str, Any]], bins: List[Dict[str, Any]], df: pl.DataFrame) -> Dict[int, Dict[str, List[str]]]:
    """
    Recommends top 5 games for each cluster based on the bins and input dataframe.

    Args:
        clusters (List[Dict[str, Any]]): List of cluster dictionaries.
        bins (List[Dict[str, Any]]): List of bin dictionaries.
        df (pl.DataFrame): Input dataframe with game_name and popularity_score.

    Returns:
        Dict[int, Dict[str, List[str]]]: A dictionary where keys are bin IDs, and values are dictionaries
        containing the games in the bin and the top 5 recommended games.
    """
    recommendations = {}

    for bin in bins:
        bin_centroid = np.array(bin['centroid'])
        best_cluster = None
        min_distance = float('inf')

        # Step 1: Find the best cluster for the bin
        for cluster in clusters:
            cluster_centroid = np.array(cluster['centroid'])
            distance = np.linalg.norm(bin_centroid - cluster_centroid)
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster

        # Step 2: Remove games in the bin from the cluster
        cluster_games = set(best_cluster['games'])
        bin_games = set(bin['games'])
        remaining_games = list(cluster_games - bin_games)

        # Step 3: Get popularity scores for the remaining games
        remaining_games_df = df.filter(pl.col("game_name").is_in(remaining_games))
        popularity_scores = remaining_games_df.select(["game_name", "popularity_score"]).to_dict(as_series=False)

        # Step 4: Sort by popularity and get top 5
        sorted_games = sorted(
            zip(popularity_scores["game_name"], popularity_scores["popularity_score"]),
            key=lambda x: x[1],
            reverse=True
        )
        top_5 = [game[0] for game in sorted_games[:5]]

        # Store recommendations for the bin
        recommendations[bin['bin_id']] = {
            "games_in_bin": list(bin_games),  # Games that made part of the bin
            "recommended_games": top_5  # Top 5 recommended games
        }

    return recommendations