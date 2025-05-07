# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import polars as pl

from .configs import *
from .utils import *

# TO DO:
# - Currently have a single distance matrix for all the clusters, can have 1 for each cluster and then only use the one needed for the current recommendation.
#   - It will be faster.
#   - It will require less memory to store it.
# - I can also improve the matrix using indexes and some function from sklearn, unless I use something like CUDA i can't probably beat the performance of it.
# - Work with Expansions/Variations.
# - Cache cluster lookups.
# - Cache previous recommendations.

class FeatureProcessor:
    """
    Processes and extracts features from game data.
    
    This class handles feature column identification and extraction from
    the game dataframes.
    """
    def extract_features(self, df: pl.DataFrame, game_name: str) -> np.ndarray:
        """
        Extracts feature vector for a specific game.
        
        Args:
            df: Dataframe containing game data.
            game_name: Name of the game to extract features for.
            
        Returns:
            NumPy array containing the game's feature values.
            
        Raises:
            ValueError: If the game is not found in the dataframe.
        """
        feature_columns = get_feature_columns(df, RELEVANT_COLUMNS)
        
        input_game_row = df.filter(pl.col("game_name") == game_name)
        if input_game_row.height == 0:
            raise ValueError(f"Game '{game_name}' not found.")
        return np.array([input_game_row[0, col] for col in feature_columns])
    
    def find_game_cluster(self, clusters: Dict[int, Dict[str, Any]], game_name: str) -> Optional[int]:
        """
        Finds the cluster ID for a given game name.
        
        Args:
            clusters: Dictionary of clusters with format 
                      {id: {'constraint': str, 'game_names': List[str], 'count': int}}
            game_name: Name of the game to find.
            
        Returns:
            Cluster ID if found, None otherwise.
        """
        for cluster_id, cluster_data in clusters.items():
            if game_name in cluster_data['game_names']:
                return cluster_id
        return None
    
    def get_candidate_games(self, clusters: Dict[int, Dict[str, Any]], cluster_id: int, game_name: str, df: pl.DataFrame) -> pl.DataFrame:
        """
        Get candidate games from the same cluster as the input game.
        
        Args:
            clusters: Dictionary of clusters with format 
                      {id: {'constraint': str, 'game_names': List[str], 'count': int}}
            cluster_id: ID of the cluster containing the input game.
            game_name: Name of the input game.
            df: Dataframe containing game data.
            
        Returns:
            DataFrame containing candidate games.
            
        Raises:
            ValueError: If no candidate games are found.
        """
        cluster_games = set(clusters[cluster_id]['game_names'])
        candidate_games = cluster_games - {game_name}
        
        if not candidate_games:
            raise ValueError("No other games found in the same cluster")
            
        candidates_df = df.filter(pl.col("game_name").is_in(candidate_games))
        
        if candidates_df.height == 0:
            raise ValueError("No candidate games found in the dataframe")
            
        return candidates_df

class ScoreCalculator:
    """
    Calculates and normalizes different types of scores for game recommendations.
    """
    
    def __init__(self):
        pass

    def precalculate_all_distances(self, df: pl.DataFrame, feature_columns: List[str], clusters: Dict[int, List[str]]) -> Dict[int, Dict[str, Union[np.ndarray, List[str]]]]:
        """
        Precalculates distances between games within each cluster.

        Args:
            df: Input DataFrame containing game features
            feature_columns: List of columns to use for distance calculation
            clusters: Dictionary mapping cluster IDs to lists of game names

        Returns:
            Dictionary mapping cluster IDs to distance matrices and game lists
        """
        cluster_distances = {}

        for cluster, games in clusters.items():
            cluster_df = df.filter(pl.col('game_name').is_in(games['game_names'])).sort('game_name')
            X = cluster_df.select(feature_columns).to_numpy()
            distances = euclidean_distances(X)
            cluster_distances[cluster] = {
                'distances': distances,
                'games': cluster_df.get_column('game_name').to_list()
            }
        return cluster_distances
       
    def calculate_rating_quality_score(self) -> pl.Expr:
        """
        Calculates Wilson lower bound score for ratings.
        
        Returns:
            Polars expression for the rating quality score calculation.
        """
        num_ratings = pl.col("num_rates")
        normalized_avg_rating = pl.col("avg_rating") / 10
        z_score = 1.96
        numerator = (
            normalized_avg_rating 
            + (z_score**2 / (2 * num_ratings)) 
            + z_score * ((normalized_avg_rating * (1 - normalized_avg_rating) / num_ratings 
                          + z_score**2 / (4 * num_ratings**2)).sqrt())
        )

        denominator = 1 + z_score**2 / num_ratings

        return (numerator / denominator).alias("rating_quality_score")
    
    def normalize_scores(self, df: pl.DataFrame, score_columns: List[str]) -> pl.DataFrame:
        """
        Normalizes score columns to 0-1 range.
        
        Args:
            df: DataFrame containing scores to normalize.
            score_columns: List of column names to normalize.
            
        Returns:
            DataFrame with additional normalized score columns.
        """
        normalization_expressions = []

        for column in score_columns:
            min_value = df[column].min()
            max_value = df[column].max()

            if max_value != min_value:
                normalized_expr = (
                    (pl.col(column) - pl.lit(min_value)) / pl.lit(max_value - min_value)
                ).alias(f"{column}_normalized")
            else:
                normalized_expr = pl.lit(1.0).alias(f"{column}_normalized")

            normalization_expressions.append(normalized_expr)

        return df.with_columns(normalization_expressions)
    
    def calculate_final_scores(self) -> pl.Expr:
        """
        Calculates the final weighted score based on normalized component scores.
        
        Args:
            df: DataFrame with normalized scores.
            
        Returns:
            Polars expression for the final score calculation.
        """
        return (
            (pl.col("popularity_score_normalized") * pl.lit(POPULARITY_WEIGHT)) +
            ((1 - pl.col("distance_score_normalized")) * pl.lit(DISTANCE_WEIGHT)) +
            (pl.col("rating_quality_score_normalized") * pl.lit(RATING_QUALITY_WEIGHT))
        ).alias("final_score")

class ExplanationGenerator:
    """
    Generates human-readable explanations for game recommendations.
    """
    
    def explain_recommendation(self, game_scores: Dict[str, float]) -> str:
        """
        Generates human-readable explanation for a recommendation.
        
        Args:
            game_scores: Dictionary of normalized scores for a recommended game.
            
        Returns:
            String containing a human-readable explanation.
        """
        explanations = []
        
        # Always explain the similarity first
        if 1 - game_scores["distance"] > 0.8:
            explanations.append("This game is very similar to your selected game")
        elif 1 - game_scores["distance"] > 0.6:
            explanations.append("This game shares many characteristics with your selected game")
        elif 1 - game_scores["distance"] > 0.4:
            explanations.append("This game has some similarities to your selected game")
        
        # Only mention popularity and ratings as secondary factors
        if game_scores["popularity"] > 0.7 and len(explanations) > 0:
            explanations.append("it's also popular among board game enthusiasts")
            
        if game_scores["rating_quality"] > 0.7 and len(explanations) > 0:
            explanations.append("it's well-rated by the community")
        
        return " and ".join(explanations) + "."


class RecommendationEngine:
    """
    Main engine for generating board game recommendations.
    
    This class orchestrates the recommendation process by combining feature processing,
    score calculation, and explanation generation.
    """
    
    def __init__(self, df: pl.DataFrame, clusters: Dict[int, Dict[str, Any]]):
        self.data = df
        self.clusters = clusters
        self.feature_columns = get_feature_columns(df, RELEVANT_COLUMNS)
        self.feature_processor = FeatureProcessor()
        self.score_calculator = ScoreCalculator()
        self.explanation_generator = ExplanationGenerator()

        print("Pre-calculating distance matrix...")
        self.distance_matrix = self.score_calculator.precalculate_all_distances(
            self.data, self.feature_columns, self.clusters
        )
        print("Distance matrix calculated.")

    def get_precalculated_distances(self, input_game: str, cluster_id: int, candidate_games: pl.DataFrame) -> List[float]:
        """
        Get distances between input game and candidate games within the same cluster.

        Args:
            cluster_id: ID of the cluster containing the games
            input_game: Name of the reference game
            candidate_games: List of game names to get distances for. If None, uses all games in cluster.

        Returns:
            List of distances in the same order as candidate_games
        """
        cluster_data = self.distance_matrix[cluster_id]
        try:
            game_idx = cluster_data['games'].index(input_game)
            distances = []
            for game in candidate_games.get_column('game_name'):
                try:
                    print(candidate_games.get_column('game_name'))
                    candidate_idx = cluster_data['games'].index(game)
                    distances.append(cluster_data['distances'][game_idx][candidate_idx])
                except ValueError:
                    distances.append(None)

            return distances
        except ValueError:
            return None
        
    def recommend_games(self, game_name: str) -> Dict[str, Any]:
        """
        Recommends games similar to the specified game.

        Args:
            clusters: Dictionary of clusters with format 
                     {id: {'constraint': str, 'game_names': List[str], 'count': int}}
            game_name: Name of the game to base recommendations on.
            df: DataFrame containing game data.

        Returns:
            Dictionary with recommended games and their scores.
        """
        try:
            target_cluster_id = self.feature_processor.find_game_cluster(self.clusters, game_name)
            if target_cluster_id is None:
                raise ValueError(f"Game '{game_name}' not found in any cluster")

            candidates_df = self.feature_processor.get_candidate_games(
                self.clusters, target_cluster_id, game_name, self.data
            )

            # Filter out games that are variations/expansions of the input game
            base_game_name = game_name.lower().strip()
            candidates_df = candidates_df.filter(
                ~pl.col("game_name").str.to_lowercase().str.starts_with(base_game_name + ":") 
            )

            candidates_df = candidates_df.with_columns([
                self.score_calculator.calculate_rating_quality_score()
            ])

            distances = self.get_precalculated_distances(
                game_name, 
                target_cluster_id,
                candidates_df
            )

            candidates_df = candidates_df.with_columns(
                pl.Series("distance_score", distances)
            )

            score_columns = [
                "popularity_score",
                "rating_quality_score",
                "distance_score"
            ]

            normalized_df = self.score_calculator.normalize_scores(candidates_df, score_columns)

            normalized_df = normalized_df.filter(
                pl.col("distance_score_normalized") <= MIN_DISTANCE_THRESHOLD
            )

            if normalized_df.height == 0:
                raise ValueError(f"No games similar enough to '{game_name}' were found")

            final_score = self.score_calculator.calculate_final_scores()

            top_5_df = (
                normalized_df
                .with_columns(final_score)
                .sort("final_score", descending=True)
                .head(5)
            )
            result = {
                "input_game": game_name,
                "cluster_id": target_cluster_id,
                "recommended_games": top_5_df["game_name"].to_list(),
                "recommendation_scores": {
                    row["game_name"]: {
                        "final_score": row["final_score"],
                        "popularity": row["popularity_score_normalized"],
                        "distance": row["distance_score_normalized"],
                        "rating_quality": row["rating_quality_score_normalized"]
                    }
                    for row in top_5_df.iter_rows(named=True)
                }
            }
            print(result)

            return result

        except ValueError as e:
            return {"error": str(e)}
    
    def explain_recommendation(self, game_scores: Dict[str, float]) -> str:
        """
        Generates human-readable explanation for a recommendation.
        
        Args:
            game_scores: Dictionary of normalized scores for a recommended game.
            
        Returns:
            String containing a human-readable explanation.
        """
        return self.explanation_generator.explain_recommendation(game_scores)