import polars as pl
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class RecommendationConfig:
    # Feature weights (rebalanced without diversity)
    POPULARITY_WEIGHT: float = 0.30
    DISTANCE_WEIGHT: float = 0.35
    RATING_QUALITY_WEIGHT: float = 0.25
    RECENCY_WEIGHT: float = 0.10
    
    # Quality thresholds
    MIN_RATINGS: int = 100
    MIN_RATING: float = 6.0
    NOVELTY_PENALTY_YEARS: int = 5
    
    # Column configuration
    COLUMN_PREFIXES: List[str] = field(default_factory=lambda: [
        "AGE_GROUP",
        "GAME_CAT",
        "LANGUAGE_DEPENDENCY",
        "GAME_DURATION"
    ])

class BoardGameRecommender:
    def __init__(self, config: RecommendationConfig = RecommendationConfig()):
        self.config = config
        
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Gets all feature columns based on prefixes."""
        feature_columns = []
        for prefix in self.config.COLUMN_PREFIXES:
            matching_cols = [col for col in df.columns if col.startswith(prefix)]
            feature_columns.extend(matching_cols)
        return feature_columns
    
    def calculate_rating_quality_score(self, df: pl.DataFrame) -> pl.Expr:
        """Calculates Wilson lower bound score for ratings."""
        N = pl.col("num_rates")
        p = pl.col("avg_rating") / 10  # Normalize to 0-1 range
        z = 1.96  # 95% confidence interval
        
        # Wilson score calculation
        numerator = (p + (z * z / (2 * N)) + 
                    z * ((p * (1 - p) / N + z * z / (4 * N * N)).sqrt()))
        denominator = 1 + z * z / N
        
        return (numerator / denominator).alias("rating_quality_score")
    
    def calculate_recency_score(self, df: pl.DataFrame) -> pl.Expr:
        """Calculates recency score with classic game consideration."""
        current_year = 2025
        return (
            1 - (
                (pl.lit(current_year) - pl.col("publication_year")) /
                pl.lit(self.config.NOVELTY_PENALTY_YEARS * 2)
            ).clip(0, 1) * 0.3
        ).alias("recency_score")
    
    def normalize_scores(
        self,
        df: pl.DataFrame,
        score_columns: List[str]
    ) -> pl.DataFrame:
        """Normalizes score columns to 0-1 range."""
        expr_list = []
        for col in score_columns:
            max_val = df[col].max()
            min_val = df[col].min()
            if max_val != min_val:
                expr = (
                    ((pl.col(col) - pl.lit(min_val)) / pl.lit(max_val - min_val))
                    .alias(f"{col}_normalized")
                )
            else:
                expr = pl.lit(1.0).alias(f"{col}_normalized")
            expr_list.append(expr)
        
        return df.with_columns(expr_list)

    def recommend_games(
        self,
        clusters: List[Dict[str, Any]],
        bins: List[Dict[str, Any]],
        df: pl.DataFrame
    ) -> Dict[int, Dict[str, Any]]:
        """
        Enhanced recommendation system that combines bin-based approach with
        sophisticated scoring.
        """
        feature_columns = self.get_feature_columns(df)
        recommendations = {}
        
        for bin in bins:
            bin_centroid = np.array(bin['centroid'])
            
            # Find best cluster for the bin
            best_cluster = min(
                clusters,
                key=lambda c: np.linalg.norm(bin_centroid - np.array(c['centroid']))
            )
            
            # Get candidate games (cluster games minus bin games)
            bin_games = set(bin['games'])
            candidate_games = set(best_cluster['games']) - bin_games
            
            if not candidate_games:
                continue
                
            # Get candidate games dataframe
            candidates_df = (
                df.filter(pl.col("game_name").is_in(candidate_games))
                .filter(pl.col("num_rates") >= self.config.MIN_RATINGS)
                .filter(pl.col("avg_rating") >= self.config.MIN_RATING)
            )
            
            if candidates_df.height == 0:
                continue
                
            # Calculate all scores
            candidates_df = candidates_df.with_columns([
                self.calculate_rating_quality_score(candidates_df),
                self.calculate_recency_score(candidates_df)
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

    def explain_recommendation(
        self,
        game_scores: Dict[str, float],
        bin_games: List[str]
    ) -> str:
        """Generates human-readable explanation for a recommendation."""
        explanations = []
        
        if game_scores["similarity"] > 0.8:
            explanations.append(
                "This game is very similar to others in your selected group"
            )
        elif game_scores["similarity"] > 0.6:
            explanations.append(
                "This game shares some key characteristics with your selected games"
            )
            
        if game_scores["popularity"] > 0.8:
            explanations.append("It's highly popular among board game enthusiasts")
            
        if game_scores["rating_quality"] > 0.8:
            explanations.append("It's very well-rated by the community")
            
        if game_scores["recency"] > 0.8:
            explanations.append("It's a recent release that's gaining attention")
            
        return " and ".join(explanations) + "."