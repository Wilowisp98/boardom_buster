import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class RecommendationConfig:
    """
    Configuration settings for the board game recommendation system.
    
    Attributes:
        POPULARITY_WEIGHT: Weight factor for popularity in the final score.
        DISTANCE_WEIGHT: Weight factor for similarity in the final score.
        RATING_QUALITY_WEIGHT: Weight factor for rating quality in the final score.
        MIN_SIMILARITY_THRESHOLD: Minimum threshold for similarity to consider a game.
        RELEVANT_COLUMNS: Feature column prefixes to include in similarity calculations.
    """
    # Feature weights
    POPULARITY_WEIGHT: float = 0.25
    DISTANCE_WEIGHT: float = 0.55
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


class FeatureProcessor:
    """
    Processes and extracts features from game data.
    
    This class handles feature column identification and extraction from
    the game dataframes.
    """
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
    
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Gets all feature columns based on configured prefixes.
        
        Args:
            df: Dataframe containing game data.
            
        Returns:
            List of column names matching the configured prefixes.
        """
        feature_columns = []
        for prefix in self.config.RELEVANT_COLUMNS:
            matching_cols = [col for col in df.columns if col.startswith(prefix)]
            feature_columns.extend(matching_cols)
        return feature_columns
    
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
        feature_columns = self.get_feature_columns(df)
        
        input_game_row = df.filter(pl.col("game_name") == game_name)
        if input_game_row.height == 0:
            raise ValueError(f"Game '{game_name}' not found in the dataframe")
            
        return np.array([input_game_row[0, col] for col in feature_columns])
    
    def find_game_cluster(
        self, 
        clusters: Dict[int, Dict[str, Any]], 
        game_name: str
    ) -> Optional[int]:
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
            clusters: Dictionary of clusters.
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
    
    This class handles various score calculations including similarity scores,
    rating quality scores, and final combined scores.
    """
    
    def __init__(self, config: RecommendationConfig):
        """
        Initialize the score calculator.
        
        Args:
            config: Configuration settings for recommendations.
        """
        self.config = config
    
    def calculate_rating_quality_score(self) -> pl.Expr:
        """
        Calculates Wilson lower bound score for ratings.
        
        Returns:
            Polars expression for the rating quality score calculation.
        """

        num_ratings = pl.col("num_rates")

        # Normalized average rating (scaled to a 0-1 range)
        normalized_avg_rating = pl.col("avg_rating") / 10

        # Z-score for a 95% confidence interval
        z_score = 1.96

        # Wilson score calculation
        numerator = (
            normalized_avg_rating 
            + (z_score**2 / (2 * num_ratings)) 
            + z_score * ((normalized_avg_rating * (1 - normalized_avg_rating) / num_ratings 
                          + z_score**2 / (4 * num_ratings**2)).sqrt())
        )

        denominator = 1 + z_score**2 / num_ratings

        return (numerator / denominator).alias("rating_quality_score")
    
    def calculate_distance_scores(self, candidates_df: pl.DataFrame, input_features: np.ndarray, feature_columns: List[str]) -> List[float]:
        """
        Calculates similarity scores based on feature distance.
        
        Args:
            candidates_df: DataFrame containing candidate games.
            input_features: Feature vector of the input game.
            feature_columns: List of feature column names.
            
        Returns:
            List of distance-based similarity scores.
        """
        distances = []
        for row in candidates_df.iter_rows(named=True):
            game_features = np.array([row[col] for col in feature_columns])
            distance = np.linalg.norm(input_features - game_features)
            distances.append(1 / (1 + distance))
            
        return distances
    
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
            (pl.col("popularity_score_normalized") * pl.lit(self.config.POPULARITY_WEIGHT)) +
            (pl.col("distance_score_normalized") * pl.lit(self.config.DISTANCE_WEIGHT)) +
            (pl.col("rating_quality_score_normalized") * pl.lit(self.config.RATING_QUALITY_WEIGHT))
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


class RecommendationEngine:
    """
    Main engine for generating board game recommendations.
    
    This class orchestrates the recommendation process by combining feature processing,
    score calculation, and explanation generation.
    """
    
    def __init__(self):
        """
        Initialize the recommendation engine.
        
        Args:
            config: Configuration settings for recommendations. If None, uses defaults.
        """
        self.config = RecommendationConfig()
        self.feature_processor = FeatureProcessor(self.config)
        self.score_calculator = ScoreCalculator(self.config)
        self.explanation_generator = ExplanationGenerator()
    
    def recommend_games(self, clusters: Dict[int, Dict[str, Any]], game_name: str, df: pl.DataFrame) -> Dict[str, Any]:
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
            # Find the cluster containing the input game
            target_cluster_id = self.feature_processor.find_game_cluster(clusters, game_name)
            if target_cluster_id is None:
                return {"error": f"Game '{game_name}' not found in any cluster"}
                
            # Get feature columns
            feature_columns = self.feature_processor.get_feature_columns(df)
            
            # Extract features for the input game
            input_game_features = self.feature_processor.extract_features(df, game_name)
            
            # Get candidate games from the same cluster
            candidates_df = self.feature_processor.get_candidate_games(
                clusters, target_cluster_id, game_name, df
            )
            
            # Calculate all scores
            candidates_df = candidates_df.with_columns([
                self.score_calculator.calculate_rating_quality_score()
            ])
            
            # Calculate distances
            distances = self.score_calculator.calculate_distance_scores(
                candidates_df, input_game_features, feature_columns
            )
            
            candidates_df = candidates_df.with_columns(
                pl.Series("distance_score", distances)
            )
            
            # Normalize all scores
            score_columns = [
                "popularity_score",
                "rating_quality_score",
                "distance_score"
            ]
            
            normalized_df = self.score_calculator.normalize_scores(candidates_df, score_columns)
            
            # Filter out games below minimum similarity threshold
            normalized_df = normalized_df.filter(
                pl.col("distance_score_normalized") >= self.config.MIN_SIMILARITY_THRESHOLD
            )
            
            # If no games meet the threshold, return a message
            if normalized_df.height == 0:
                return {"error": f"No games similar enough to '{game_name}' were found"}
                
            # Calculate final score
            final_score = self.score_calculator.calculate_final_scores()
            
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