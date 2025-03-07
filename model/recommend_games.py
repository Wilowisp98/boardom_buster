import polars as pl
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class RecommendationConfig:
    # Feature weights - increased similarity importance
    POPULARITY_WEIGHT: float = 0.25
    DISTANCE_WEIGHT: float = 0.55  # Increased from 0.39
    RATING_QUALITY_WEIGHT: float = 0.20
    
    # Minimum threshold for similarity score
    MIN_SIMILARITY_THRESHOLD: float = 0.4
    
    # Column configuration
    RELEVANT_COLUMNS: List[str] = field(default_factory=lambda: [
        "AGE_GROUP",
        "GAME_CAT",
        "LANGUAGE_DEPENDENCY",
        "GAME_DURATION",
        "GAME_DIFFICULTY"
    ])

class BoardGameRecommendation:
    def __init__(self):
        self.config = RecommendationConfig()
        
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Gets all feature columns based on prefixes.
        """
        feature_columns = []
        for prefix in self.config.RELEVANT_COLUMNS:
            matching_cols = [col for col in df.columns if col.startswith(prefix)]
            feature_columns.extend(matching_cols)
        return feature_columns
    
    def calculate_rating_quality_score(self) -> pl.Expr:
        """
        Calculates Wilson lower bound score for ratings.
        """
        # Number of ratings
        num_ratings = pl.col("num_rates")

        # Normalized average rating (scaled to a 0-1 range)
        normalized_avg_rating = pl.col("avg_rating") / 10

        # Z-score for a 95% confidence interval (1.96 corresponds to 95% confidence)
        z_score = 1.96

        # Wilson score calculation
        # Numerator: adjusted average rating with confidence interval
        numerator = (
            normalized_avg_rating 
            + (z_score**2 / (2 * num_ratings)) 
            + z_score * ((normalized_avg_rating * (1 - normalized_avg_rating) / num_ratings 
                          + z_score**2 / (4 * num_ratings**2)).sqrt())
        )

        # Denominator: scaling factor for the confidence interval
        denominator = 1 + z_score**2 / num_ratings

        return (numerator / denominator).alias("rating_quality_score")
    
    def normalize_scores(self, df: pl.DataFrame, score_columns: List[str]) -> pl.DataFrame:
        """
        Normalizes score columns to 0-1 range.
        """
        # Initialize a list to store normalization expressions
        normalization_expressions = []

        for column in score_columns:
            # Calculate the minimum and maximum values of the column
            min_value = df[column].min()
            max_value = df[column].max()

            # Check if the column has a valid range (min != max)
            if max_value != min_value:
                # Normalize the column to a 0-1 range
                normalized_expr = (
                    (pl.col(column) - pl.lit(min_value)) / pl.lit(max_value - min_value)
                ).alias(f"{column}_normalized")
            else:
                # If min == max, set the normalized value to 1.0 (or 0.0, depending on your use case)
                normalized_expr = pl.lit(1.0).alias(f"{column}_normalized")

            # Add the normalization expression to the list
            normalization_expressions.append(normalized_expr)

        # Apply all normalization expressions to the DataFrame
        return df.with_columns(normalization_expressions)

    def recommend_games(self, clusters: Dict[int, Dict[str, Any]], game_name: str, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Recommendation system based on a single game name

        Args:
            clusters: Dictionary of clusters with format {id: {'constraint': str, 'game_names': List[str], 'count': int}}
            game_name: Name of the game to base recommendations on
            df: DataFrame containing game data

        Returns:
            Dictionary with recommended games and their scores
        """
        feature_columns = self.get_feature_columns(df)

        # Find which cluster the input game belongs to
        target_cluster_id = None
        for cluster_id, cluster_data in clusters.items():
            if game_name in cluster_data['game_names']:
                target_cluster_id = cluster_id
                break
            
        if target_cluster_id is None:
            return {"error": f"Game '{game_name}' not found in any cluster"}

        # Get candidate games (all games in the cluster except the input game)
        cluster_games = set(clusters[target_cluster_id]['game_names'])
        candidate_games = cluster_games - {game_name}

        if not candidate_games:
            return {"error": "No other games found in the same cluster"}

        # Get candidate games dataframe
        candidates_df = df.filter(pl.col("game_name").is_in(candidate_games))

        if candidates_df.height == 0:
            return {"error": "No candidate games found in the dataframe"}

        # Get feature vector for the input game
        input_game_row = df.filter(pl.col("game_name") == game_name)
        if input_game_row.height == 0:
            return {"error": f"Game '{game_name}' not found in the dataframe"}

        # Extract features for the input game
        input_game_features = np.array([input_game_row[0, col] for col in feature_columns])

        # Calculate all scores
        candidates_df = candidates_df.with_columns([
            self.calculate_rating_quality_score()
        ])

        # Calculate distances
        distances = []
        for row in candidates_df.iter_rows(named=True):
            game_features = np.array([row[col] for col in feature_columns])
            distance = np.linalg.norm(input_game_features - game_features)
            distances.append(1 / (1 + distance))  # Convert to similarity score

        candidates_df = candidates_df.with_columns(
            pl.Series("distance_score", distances)
        )

        # Normalize all scores
        score_columns = [
            "popularity_score",
            "rating_quality_score",
            "distance_score"
        ]

        normalized_df = self.normalize_scores(candidates_df, score_columns)
        
        # Filter out games below minimum similarity threshold
        normalized_df = normalized_df.filter(
            pl.col("distance_score_normalized") >= self.config.MIN_SIMILARITY_THRESHOLD
        )
        
        # If no games meet the threshold, return a message
        if normalized_df.height == 0:
            return {"error": f"No games similar enough to '{game_name}' were found"}

        # Calculate final score
        final_score = (
            (pl.col("popularity_score_normalized") * pl.lit(self.config.POPULARITY_WEIGHT)) +
            (pl.col("distance_score_normalized") * pl.lit(self.config.DISTANCE_WEIGHT)) +
            (pl.col("rating_quality_score_normalized") * pl.lit(self.config.RATING_QUALITY_WEIGHT))
        ).alias("final_score")

        # Get top 5 recommendations
        top_5_df = (
            normalized_df
            .with_columns(final_score)
            .sort("final_score", descending=True)
            .head(5)
        )

        # Format recommendations
        result = {
            "input_game": game_name,
            "cluster_id": target_cluster_id,
            "recommended_games": top_5_df["game_name"].to_list(),
            "recommendation_scores": {
                row["game_name"]: {
                    "final_score": row["final_score"],
                    "popularity": row["popularity_score_normalized"],
                    "similarity": row["distance_score_normalized"],
                    "rating_quality": row["rating_quality_score_normalized"]
                }
                for row in top_5_df.iter_rows(named=True)
            }
        }

        return result

    def explain_recommendation(self, game_scores: Dict[str, float]) -> str:
        """
        Generates human-readable explanation for a recommendation.
        """
        explanations = []
        
        # Always explain the similarity first
        if game_scores["similarity"] > 0.8:
            explanations.append("This game is very similar to your selected game")
        elif game_scores["similarity"] > 0.6:
            explanations.append("This game shares many characteristics with your selected game")
        elif game_scores["similarity"] > 0.4:
            explanations.append("This game has some similarities to your selected game")
        
        # Only mention popularity and ratings as secondary factors
        if game_scores["popularity"] > 0.7 and len(explanations) > 0:
            explanations.append("it's also popular among board game enthusiasts")
            
        if game_scores["rating_quality"] > 0.7 and len(explanations) > 0:
            explanations.append("it's well-rated by the community")
        
        return " and ".join(explanations) + "."