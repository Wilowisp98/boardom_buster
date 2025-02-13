import polars as pl
import html
import re
from typing import Dict, List
from datetime import datetime
import numpy as np

# Constants for mappings
AGE_MAPPING: Dict[str, int] = {"Unknown": 0, "Kids": 1, "Family": 2, "Teen": 3, "Adult": 4}
PLAYING_TIME_MAPPING: Dict[str, int] = {"Unknown": 0, "Short": 1, "Medium": 2, "Long": 3, "Very Long": 4}
POPULARITY_WEIGHTS: Dict[str, float] = {
    "owned_by": 0.35,
    "wished_by": 0.15,
    "num_rates": 0.15,
    "avg_rating": 0.20,
    "recency": 0.15,
}
UNNECESSARY_COLUMNS: List[str] = [
    "best_num_players",
    "families",
    "designers",
    "artists",
    "publishers",
    "min_playtime",
    "max_playtime",
    "min_age"
]

def clean_text_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Clean text data by removing HTML entities, special characters, newlines, 
    and normalizing whitespace, replacing the original column.
    
    Args:
        df: Input Polars DataFrame
        column: Name of the text column to clean
        
    Returns:
        DataFrame with the cleaned column replacing the original
    
    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "text": ["This &amp; that &mdash; testing", "Another &quot;test&quot; string"]
        ... })
        >>> clean_text_column(df, "text")
        shape: (2, 2)
        ┌─────────────────────────────────┬────────────────────────┐
        │ text                            ┆ text_cleaned           │
        │ ---                             ┆ ---                    │
        │ str                             ┆ str                    │
        ╞═════════════════════════════════╪════════════════════════╡
        │ This &amp; that &mdash; testing ┆ This and that testing  │
        │ Another &quot;test&quot; string ┆ Another test string    │
        └──────────────────────────────── ┴────────────────────────┘
    """
    # Replace all whitespace (including \n) with single space
    # Convert HTML entities
    # Remove remaining &
    # Remove -
    # Remove backslashes
    # Remove quotes
    # Remove leading/trailing whitespace
    try:
        cleaned_col = (
            pl.col(column)
            .map_elements(
                lambda x: re.sub(r'\s+', ' ', html.unescape(str(x)))
                .translate(str.maketrans("", "", '&-"\\'))
                .strip()
                if x is not None else None,
                return_dtype=pl.String
            )
            .alias(column)
        )

        return df.with_columns(cleaned_col)
    except Exception as e:
        raise ValueError(f"Error cleaning {column}: {e}")

def add_popularity_score(df: pl.DataFrame, decay_half_life_days: int = 365) -> pl.DataFrame:
    """
    Add a popularity score column to a games DataFrame based on ownership, wishlists, ratings count,
    and average rating. The score is normalized between 0 and 1.

    Args:
        df (pl.DataFrame): DataFrame containing game metrics columns: owned_by, wished_by,
            num_rates, and avg_rating.

    Returns:
        pl.DataFrame: Original DataFrame with an additional 'popularity_score' column.
    """
    def percentile_normalize(series: pl.Series) -> pl.Series:
        """Convert values to percentiles (0-1 range) to handle outliers better"""
        return series.rank() / len(series)
    
    def log_normalize(series: pl.Series) -> pl.Series:
        """Apply log normalization for metrics with high variance"""
        return pl.Series(np.log1p(series.to_numpy())) / pl.Series(np.log1p(series.max()))
    
    def calculate_time_decay(publication_year: pl.Series) -> pl.Series:
        """Calculate time-based decay factor using publication year"""
        current_year = datetime.now().year
        years_since_publication = current_year - publication_year
        days_since_publication = years_since_publication * 365  # Approximate days
        decay = 2 ** (-days_since_publication / decay_half_life_days)
        return decay

    try:
        # Create normalized columns using different strategies
        normalized_cols = [
            log_normalize(pl.col("owned_by")).alias("owned_by_norm"),
            log_normalize(pl.col("wished_by")).alias("wished_by_norm"),
            log_normalize(pl.col("num_rates")).alias("num_rates_norm"),
            percentile_normalize(pl.col("avg_rating")).alias("avg_rating_norm"),
            calculate_time_decay(pl.col("publication_year")).alias("recency_norm")
        ]

        # Add normalized columns
        df_with_norms = df.with_columns(normalized_cols)

        # Calculate final weighted score
        popularity_score = (
            (pl.col("owned_by_norm") * POPULARITY_WEIGHTS["owned_by"]) +
            (pl.col("wished_by_norm") * POPULARITY_WEIGHTS["wished_by"]) +
            (pl.col("num_rates_norm") * POPULARITY_WEIGHTS["num_rates"]) +
            (pl.col("avg_rating_norm") * POPULARITY_WEIGHTS["avg_rating"]) +
            (pl.col("recency_norm") * POPULARITY_WEIGHTS["recency"])
        ).alias("popularity_score")

        return df_with_norms.with_columns(popularity_score)
    except Exception as e:
        raise ValueError(f"Error calculating improved popularity score: {e}")

def one_hot_encode(df: pl.DataFrame, columns: List[str], prefix: str = None) -> pl.DataFrame:
    """
    Perform one-hot encoding on one or multiple categorical columns that may contain either single values
    or lists of values. Creates binary indicator variables for each unique category across all specified columns.

    Args:
        df (pl.DataFrame): Input DataFrame containing categorical columns.
        columns (List[str]): List of categorical column names to be one-hot encoded.
        prefix (str, optional): Prefix to add to the generated column names.

    Returns:
        pl.DataFrame: Original DataFrame with additional binary columns for one-hot encoding.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B", "Game C"],
        ...     "tags": [["Strategy", "Economic"], ["Party", "Strategy"], ["Strategy"]]
        ... })
        >>> df_encoded = one_hot_encode(df, ["tags"])
        >>> print(df_encoded)
        shape: (3, 5)
        ┌───────────┬─────────────────────┬──────────────┬────────────┬────────────────┐
        │ game_name │ tags                │ tag_Strategy │ tag_Party  │ tag_Economic   │
        │ ---       │ ---                 │ ---          │ ---        │ ---            │
        │ str       │ list[str]           │ u8           │ u8         │ u8             │
        ╞═══════════╪═════════════════════╪══════════════╪════════════╪════════════════╡
        │ Game A    │ ["Strategy", "Eco...│ 1            │ 0          │ 1              │
        │ Game B    │ ["Party", "Strat... │ 1            │ 1          │ 0              │
        │ Game C    │ ["Strategy"]        │ 1            │ 0          │ 0              │
        └───────────┴─────────────────────┴──────────────┴────────────┴────────────────┘
    """
    try:
        # Get all unique non-null values across specified columns
        all_unique_values = set()
        for column in columns:
            # Check if the column contains lists
            sample_value = df.select(pl.col(column)).row(0)[0]
            is_list_column = isinstance(sample_value, list)

            if is_list_column:
                # For list columns, explode and get unique values
                unique_values = (
                    df.select(pl.col(column))
                    .filter(pl.col(column).is_not_null())
                    .explode(column)
                    .unique()
                    .to_series()
                    .to_list()
                )
            else:
                # For single value columns, get unique values as before
                unique_values = (
                    df.select(pl.col(column))
                    .filter(pl.col(column).is_not_null())
                    .unique()
                    .to_series()
                    .to_list()
                )
            all_unique_values.update(unique_values)

        # Start with a copy of the original dataframe
        result_df = df.clone()

        # Create binary columns for each unique value
        for value in all_unique_values:
            binary_exprs = []
            for col in columns:
                sample_value = df.select(pl.col(col)).row(0)[0]
                is_list_column = isinstance(sample_value, list)

                if is_list_column:
                    # For list columns, check if value is in the list
                    expr = pl.col(col).map_elements(lambda x: value in (x or [])).fill_null(False)
                else:
                    # For single value columns, check equality
                    expr = pl.col(col).fill_null(pl.lit("__NULL__")) == value
                binary_exprs.append(expr)
            
            # Combine expressions with OR operation and cast to UInt8
            column_name = f"{prefix}_{str(value).lower()}" if prefix else f"{str(value).lower()}"
            result_df = result_df.with_columns(
                pl.any_horizontal(binary_exprs)
                .cast(pl.UInt8)
                .alias(column_name)
            )

        return result_df
    except Exception as e:
        raise ValueError(f"Error performing one-hot encoding: {str(e)}")

def normalize_play_age(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the suggested play age into broader categories: Kids, Family, Teen, and Adult.

    Args:
        df (pl.DataFrame): Input DataFrame with the 'suggested_play_age' column.

    Returns:
        pl.DataFrame: Updated DataFrame with an additional 'age_group' column.

    Example:
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B", "Game C", "Game D"],
        ...     "suggested_play_age": [6, 10, 14, 18]
        ... })
        >>> df_normalized = normalize_play_age(df)
        >>> print(df_normalized)
        shape: (4, 3)
        ┌───────────┬────────────────────┬───────────┐
        │ game_name │ suggested_play_age ┆ age_group │
        │ ---       │ ---                │ ---       │
        │ str       │ i64                │ str       │
        ╞═══════════╪════════════════════╪═══════════╡
        │ Game A    │ 6                  │ Kids      │
        │ Game B    │ 10                 │ Family    │
        │ Game C    │ 14                 │ Teen      │
        │ Game D    │ 18                 │ Adult     │
        └───────────┴────────────────────┴───────────┘
    """
    try:
        age_group_col = (
            pl.when(pl.col("suggested_play_age") <= 8).then(pl.lit("Kids"))
            .when((pl.col("suggested_play_age") > 8) & (pl.col("suggested_play_age") <= 12)).then(pl.lit("Family"))
            .when((pl.col("suggested_play_age") > 12) & (pl.col("suggested_play_age") <= 16)).then(pl.lit("Teen"))
            .when(pl.col("suggested_play_age") > 16).then(pl.lit("Adult"))
            .otherwise(pl.lit("Unknown"))
            .alias("age_group")
        )
        return df.with_columns(age_group_col)
    except Exception as e:
        raise ValueError(f"Error normalizing play age: {e}")

def normalize_playing_time(df: pl.DataFrame) -> pl.DataFrame:
    """
    Categorize the playing time into buckets based on the duration.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: Updated DataFrame with categorized playing time.

    Example:
        >>> df = pl.DataFrame({"playing_time": [15, 45, 75, 140]})
        >>> categorize_playing_time(df)
        shape: (4, 2)
        ┌──────────────┬────────────────────────┐
        │ playing_time ┆ playing_time_group_col │
        │ ---          ┆ ---                    │
        │ i64          ┆ str                    │
        ╞══════════════╪════════════════════════╡
        │ 15           ┆ Short                  │
        │ 45           ┆ Medium                 │
        │ 75           ┆ Long                   │
        │ 140          ┆ Very Long              │
        └──────────────┴────────────────────────┘
    """
    try:
        playing_time_group_col = (
            pl.when(pl.col("playing_time") <= 30).then(pl.lit("Short"))
            .when((pl.col("playing_time") > 30) & (pl.col("playing_time") <= 60)).then(pl.lit("Medium"))
            .when((pl.col("playing_time") > 60) & (pl.col("playing_time") <= 120)).then(pl.lit("Long"))
            .when(pl.col("playing_time") > 120).then(pl.lit("Very_Long"))
            .otherwise(pl.lit("Unknown"))
            .alias("playing_time_group")
        )
        return df.with_columns(playing_time_group_col)
    except Exception as e:
        raise ValueError(f"Error categorizing playing time: {e}")

def normalize_player_count(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add binary flags for different player count categories based on min_players and max_players columns.

    Args:
        df (pl.DataFrame): Input DataFrame with 'min_players' and 'max_players' columns.

    Returns:
        pl.DataFrame: Updated DataFrame with additional player count category flags.
        
    Example:
        >>> df = pl.DataFrame({
        ...     "game_name": ["Solitaire", "Chess", "Monopoly", "Party Game"],
        ...     "min_players": [1, 2, 3, 6],
        ...     "max_players": [1, 2, 6, 12]
        ... })
        >>> normalize_player_count(df)
        shape: (4, 7)
        ┌────────────┬─────────────┬─────────────┬─────────────────┬───────────────────┬──────────────────┬──────────────────┐
        │ game_name  ┆ min_players ┆ max_players ┆ GROUP_SIZE_solo ┆ GROUP_SIZE_couple ┆ GROUP_SIZE_small ┆ GROUP_SIZE_party │
        │ ---        ┆ ---         ┆ ---         ┆ ---             ┆ ---               ┆ ---              ┆ ---              │
        │ str        ┆ i64         ┆ i64         ┆ u8              ┆ u8                ┆ u8               ┆ u8               │
        ╞════════════╪═════════════╪═════════════╪═════════════════╪═══════════════════╪══════════════════╪══════════════════╡
        │ Solitaire  ┆ 1           ┆ 1           ┆ 1               ┆ 0                 ┆ 0                ┆ 0                │
        │ Chess      ┆ 2           ┆ 2           ┆ 0               ┆ 1                 ┆ 0                ┆ 0                │
        │ Monopoly   ┆ 3           ┆ 6           ┆ 0               ┆ 0                 ┆ 1                ┆ 1                │
        │ Party Game ┆ 6           ┆ 12          ┆ 0               ┆ 0                 ┆ 0                ┆ 1                │
        └────────────┴─────────────┴─────────────┴─────────────────┴───────────────────┴──────────────────┴──────────────────┘
    """

    try:
        # Create binary feature columns
        df = df.with_columns([
            (pl.col("min_players") <= 1).cast(pl.UInt8).alias("GROUP_SIZE_solo"),
            ((pl.col("min_players") <= 2) & (pl.col("max_players") >= 2)).cast(pl.UInt8).alias("GROUP_SIZE_couple"),
            ((pl.col("min_players") >= 3) & (pl.col("max_players") <= 5)).cast(pl.UInt8).alias("GROUP_SIZE_small"),
            (pl.col("max_players") >= 6).cast(pl.UInt8).alias("GROUP_SIZE_party")
        ])
        
        return df
    except Exception as e:
        raise ValueError(f"Error normalizing player count: {str(e)}")

def run_data_preparation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Run the complete data preparation pipeline.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: Processed DataFrame.
    """
    try:
        # df = df.drop(UNNECESSARY_COLUMNS)
        df = clean_text_column(df, "description")
        df = add_popularity_score(df)
        # df = one_hot_encode(df, ['subcategory_1', 'subcategory_2'])
        df = normalize_play_age(df)
        df = one_hot_encode(df, ['age_group'], 'AGE_GROUP')
        # df = encode_column(df, "age_group", AGE_MAPPING)
        df = normalize_playing_time(df)
        df = one_hot_encode(df, ['playing_time_group'], 'GAME_DURATION')
        # df = encode_column(df, "playing_time_group", PLAYING_TIME_MAPPING)
        df = normalize_player_count(df)
        df = one_hot_encode(df, ['categories'], 'GAME_CAT')

        return df
    except Exception as e:
        raise ValueError(f"Error running data preparation: {e}")
    


# def encode_column(df: pl.DataFrame, column_name: str, mapping_dict: Dict[str, int]) -> pl.DataFrame:
#     """
#     General function to encode a categorical column using a provided mapping dictionary.
# 
#     Args:
#         df (pl.DataFrame): Input DataFrame.
#         column_name (str): Name of the column to encode.
#         mapping_dict (Dict[str, int]): Dictionary for mapping categories to numerical values.
# 
#     Returns:
#         pl.DataFrame: Updated DataFrame with the encoded column.
# 
#     Example:
#         >>> df = pl.DataFrame({"age_group": ["Kids", "Family", "Teen", "Adult"]})
#         >>> age_mapping = {"Kids": 0, "Family": 1, "Teen": 2, "Adult": 3}
#         >>> encode_column(df, "age_group", age_mapping)
#         shape: (4, 2)
#         ┌──────────┬─────────────┐
#         │ age_group│ age_encoded │
#         │ ---      │ ---         │
#         │ str      │ i64         │
#         ╞══════════╪═════════════╡
#         │ Kids     │ 0           │
#         │ Family   │ 1           │
#         │ Teen     │ 2           │
#         │ Adult    │ 3           │
#         └──────────┴─────────────┘
#     """
#     try:
#         encoded_col = pl.col(column_name).replace(mapping_dict).cast(pl.Int32).alias(f"{column_name}_encoded")
#         return df.with_columns(encoded_col)
#     except Exception as e:
#         raise ValueError(f"Error encoding column '{column_name}': {e}")