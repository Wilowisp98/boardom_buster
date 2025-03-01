import polars as pl
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class RecommendationConfig:
    # Feature weights
    POPULARITY_WEIGHT: float = 0.30
    DISTANCE_WEIGHT: float = 0.35
    RATING_QUALITY_WEIGHT: float = 0.25
    RECENCY_WEIGHT: float = 0.10
    
    # Quality thresholds
    NOVELTY_PENALTY_YEARS: int = 5
    
    # Column configuration
    RELEVANT_COLUMNS: List[str] = field(default_factory=lambda: [
        "AGE_GROUP",
        "GAME_CAT",
        "LANGUAGE_DEPENDENCY",
        "GAME_DURATION",
        "GAME_DIFFICULTY"
    ])

class BoardGameRecommendation:
    def __init__(self, config: RecommendationConfig = RecommendationConfig()):
        self.config = config
        
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
    
    def calculate_recency_score(self) -> pl.Expr:
        """
        Calculates recency score with classic game consideration.
        """
        current_year = datetime.now().year
        
        # Calculate the age of the item (in years)
        age_of_item = current_year - pl.col("publication_year")
        
        # Define the novelty penalty threshold (in years)
        novelty_penalty_years = self.config.NOVELTY_PENALTY_YEARS * 2
        
        # Calculate the recency penalty as a proportion of the novelty penalty threshold
        recency_penalty = (age_of_item / novelty_penalty_years).clip(0, 1)
        
        # Apply the recency penalty to the score (30% penalty for older items)
        recency_score = 1 - (recency_penalty * 0.3)
        
        return recency_score.alias("recency_score")
    
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

    def recommend_games(self, clusters: List[Dict[str, Any]], bins: List[Dict[str, Any]], df: pl.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Recommendation system
        """
        feature_columns = self.get_feature_columns(df)
        recommendations = {}
        
        for bin in bins:
            bin_centroid = np.array(bin['centroid'])
            
            # Find best cluster for the bin
            best_cluster = min(
                clusters,
                key=lambda cluster: np.linalg.norm(bin_centroid - np.array(cluster['centroid']))
            )
            
            # Get candidate games (cluster games minus bin games)
            bin_games = set(bin['games'])
            candidate_games = set(best_cluster['games']) - bin_games
            
            if not candidate_games:
                continue
                
            # Get candidate games dataframe
            candidates_df = df.filter(pl.col("game_name").is_in(candidate_games))
            
            if candidates_df.height == 0:
                continue
                
            # Calculate all scores
            candidates_df = candidates_df.with_columns([
                self.calculate_rating_quality_score(),
                self.calculate_recency_score()
            ])
            
            # Calculate distances
            distances = []
            for row in candidates_df.iter_rows(named=True):
                game_features = np.array([row[col] for col in feature_columns])
                distance = np.linalg.norm(bin_centroid - game_features)
                distances.append(1 / (1 + distance))  # Convert to similarity score
            
            candidates_df = candidates_df.with_columns(
                pl.Series("distance_score", distances)
            )
            
            # Normalize all scores
            score_columns = [
                "popularity_score",
                "rating_quality_score",
                "recency_score",
                "distance_score"
            ]
            
            normalized_df = self.normalize_scores(candidates_df, score_columns)
            
            # Calculate final score
            final_score = (
                (pl.col("popularity_score_normalized") * pl.lit(self.config.POPULARITY_WEIGHT)) +
                (pl.col("distance_score_normalized") * pl.lit(self.config.DISTANCE_WEIGHT)) +
                (pl.col("rating_quality_score_normalized") * pl.lit(self.config.RATING_QUALITY_WEIGHT)) +
                (pl.col("recency_score_normalized") * pl.lit(self.config.RECENCY_WEIGHT))
            ).alias("final_score")
            
            # Get top 5 recommendations
            top_5_df = (
                normalized_df
                .with_columns(final_score)
                .sort("final_score", descending=True)
                .head(5)
            )
            
            # Store recommendations
            recommendations[bin['bin_id']] = {
                "games_in_bin": list(bin_games),
                "recommended_games": top_5_df["game_name"].to_list(),
                "recommendation_scores": {
                    row["game_name"]: {
                        "final_score": row["final_score"],
                        "popularity": row["popularity_score_normalized"],
                        "similarity": row["distance_score_normalized"],
                        "rating_quality": row["rating_quality_score_normalized"],
                        "recency": row["recency_score_normalized"]
                    }
                    for row in top_5_df.iter_rows(named=True)
                }
            }
        
        return recommendations

    def explain_recommendation(self, game_scores: Dict[str, float]) -> str:
        """
        Generates human-readable explanation for a recommendation.
        """
        explanations = []
        
        if game_scores["similarity"] > 0.8:
            explanations.append("This game is very similar to others in your selected group")

        elif game_scores["similarity"] > 0.6:
            explanations.append("This game shares some key characteristics with your selected games")
            
        if game_scores["popularity"] > 0.8:
            explanations.append("It's highly popular among board game enthusiasts")
            
        if game_scores["rating_quality"] > 0.8:
            explanations.append("It's very well-rated by the community")
            
        if game_scores["recency"] > 0.8:
            explanations.append("It's a recent release that's gaining attention")
            
        return " and ".join(explanations) + "."