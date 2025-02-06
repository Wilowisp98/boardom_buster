import polars as pl
import html
import re
from typing import Dict, List

# Constants for mappings
AGE_MAPPING: Dict[str, int] = {"Unknown": 0, "Kids": 1, "Family": 2, "Teen": 3, "Adult": 4}
PLAYING_TIME_MAPPING: Dict[str, int] = {"Unknown": 0, "Short": 1, "Medium": 2, "Long": 3, "Very Long": 4}
POPULARITY_WEIGHTS: Dict[str, float] = {
    "owned_by": 0.4,
    "wished_by": 0.2,
    "num_rates": 0.2,
    "avg_rating": 0.2,
}
UNNECESSARY_COLUMNS: List[str] = [
    "publication_year",
    "min_players",
    "max_players",
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
        df (pl.DataFrame): Input Polars DataFrame.
        column (str): Name of the text column to clean.

    Returns:
        pl.DataFrame: DataFrame with the cleaned column replacing the original.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "text": ["This &amp; that &mdash; testing", "Another &quot;test&quot; string"]
        ... })
        >>> clean_text_column(df, "text")
        shape: (2, 1)
        ┌─────────────────────────────────┐
        │ text                            │
        │ ---                             │
        │ str                             │
        ╞═════════════════════════════════╡
        │ This and that testing           │
        │ Another test string             │
        └─────────────────────────────────┘
    """
    try:
        cleaned_col = (
            pl.col(column)
            .apply(lambda x: re.sub(r'\s+', ' ', html.unescape(str(x))) if x is not None else None)
            .str.replace_all(r'[&-"\\]', '')
            .str.strip()
            .alias(column)
        )
        return df.with_columns(cleaned_col)
    except Exception as e:
        raise ValueError(f"Error cleaning text column '{column}': {e}")

def add_popularity_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a popularity score column to a games DataFrame based on ownership, wishlists, ratings count,
    and average rating. The score is normalized between 0 and 1.

    Args:
        df (pl.DataFrame): DataFrame containing game metrics columns: owned_by, wished_by,
            num_rates, and avg_rating.

    Returns:
        pl.DataFrame: Original DataFrame with an additional 'popularity_score' column.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B"],
        ...     "owned_by": [1000, 500],
        ...     "wished_by": [200, 100],
        ...     "num_rates": [800, 400],
        ...     "avg_rating": [8.5, 7.5]
        ... })
        >>> add_popularity_score(df)
        shape: (2, 6)
        ┌───────────┬──────────┬──────────┬───────────┬────────────┬─────────────────┐
        │ game_name ┆ owned_by ┆ wished_by┆ num_rates ┆ avg_rating ┆ popularity_score│
        │ ---       ┆ ---      ┆ ---      ┆ ---       ┆ ---        ┆ ---             │
        │ str       ┆ i64      ┆ i64      ┆ i64       ┆ f64        ┆ f64             │
        ╞═══════════╪══════════╪══════════╪═══════════╪════════════╪═════════════════╡
        │ Game A    ┆ 1000     ┆ 200      ┆ 800       ┆ 8.5        ┆ 1.0             │
        │ Game B    ┆ 500      ┆ 100      ┆ 400       ┆ 7.5        ┆ 0.61            │
        └───────────┴──────────┴──────────┴───────────┴────────────┴─────────────────┘
    """
    try:
        popularity_col = (
            (pl.col("owned_by") / pl.col("owned_by").max() * POPULARITY_WEIGHTS["owned_by"]) +
            (pl.col("wished_by") / pl.col("wished_by").max() * POPULARITY_WEIGHTS["wished_by"]) +
            (pl.col("num_rates") / pl.col("num_rates").max() * POPULARITY_WEIGHTS["num_rates"]) +
            (pl.col("avg_rating") / 10 * POPULARITY_WEIGHTS["avg_rating"])
        ).alias("popularity_score")

        return df.with_columns(popularity_col)
    except Exception as e:
        raise ValueError(f"Error adding popularity score: {e}")

def one_hot_encode(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Perform one-hot encoding on one or multiple categorical columns by creating binary indicator variables
    for each unique category across the specified columns.

    Args:
        df (pl.DataFrame): Input DataFrame containing categorical columns.
        columns (List[str]): List of categorical column names to be one-hot encoded.

    Returns:
        pl.DataFrame: Original DataFrame with additional binary columns for one-hot encoding.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B", "Game C"],
        ...     "subcategory_1": ["Strategy", "Party", "Strategy"],
        ...     "subcategory_2": ["Economic", "Strategy", None]
        ... })
        >>> df_encoded = one_hot_encode(df, ["subcategory_1", "subcategory_2"])
        >>> print(df_encoded)
        shape: (3, 6)
        ┌───────────┬──────────────┬──────────────┬──────────┬───────┬──────────┐
        │ game_name │ subcategory_1│ subcategory_2│ Strategy │ Party │ Economic │
        │ ---       │ ---          │ ---          │ ---      │ ---   │ ---      │
        │ str       │ str          │ str          │ u8       │ u8    │ u8       │
        ╞═══════════╪══════════════╪══════════════╪══════════╪═══════╪══════════╡
        │ Game A    │ Strategy     │ Economic     │ 1        │ 0     │ 1        │
        │ Game B    │ Party        │ Strategy     │ 1        │ 1     │ 0        │
        │ Game C    │ Strategy     │ None         │ 1        │ 0     │ 0        │
        └───────────┴──────────────┴──────────────┴──────────┴───────┴──────────┘
    """
    try:
        all_unique_values = set()
        for column in columns:
            unique_values = df.select(pl.col(column)).drop_nulls().unique().to_series().to_list()
            all_unique_values.update(unique_values)

        result_df = df.clone()
        for value in all_unique_values:
            binary_exprs = [pl.col(col) == value for col in columns]
            result_df = result_df.with_columns(
                pl.any_horizontal(binary_exprs).cast(pl.UInt8).alias(f"tag_{value}")
            )
        return result_df
    except Exception as e:
        raise ValueError(f"Error performing one-hot encoding: {e}")

def encode_column(df: pl.DataFrame, column_name: str, mapping_dict: Dict[str, int]) -> pl.DataFrame:
    """
    General function to encode a categorical column using a provided mapping dictionary.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column_name (str): Name of the column to encode.
        mapping_dict (Dict[str, int]): Dictionary for mapping categories to numerical values.

    Returns:
        pl.DataFrame: Updated DataFrame with the encoded column.

    Example:
        >>> df = pl.DataFrame({"age_group": ["Kids", "Family", "Teen", "Adult"]})
        >>> age_mapping = {"Kids": 0, "Family": 1, "Teen": 2, "Adult": 3}
        >>> encode_column(df, "age_group", age_mapping)
        shape: (4, 2)
        ┌──────────┬─────────────┐
        │ age_group│ age_encoded │
        │ ---      │ ---         │
        │ str      │ i64         │
        ╞══════════╪═════════════╡
        │ Kids     │ 0           │
        │ Family   │ 1           │
        │ Teen     │ 2           │
        │ Adult    │ 3           │
        └──────────┴─────────────┘
    """
    try:
        encoded_col = pl.col(column_name).replace(mapping_dict).cast(pl.Int32).alias(f"{column_name}_encoded")
        return df.with_columns(encoded_col)
    except Exception as e:
        raise ValueError(f"Error encoding column '{column_name}': {e}")

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
            .when(pl.col("playing_time") > 120).then(pl.lit("Very Long"))
            .otherwise(pl.lit("Unknown"))
            .alias("playing_time_group")
        )
        return df.with_columns(playing_time_group_col)
    except Exception as e:
        raise ValueError(f"Error categorizing playing time: {e}")

def run_data_preparation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Run the complete data preparation pipeline.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: Processed DataFrame.
    """
    try:
        df = df.drop(UNNECESSARY_COLUMNS)
        df = clean_text_column(df, "description")
        df = add_popularity_score(df)
        df = one_hot_encode(df, ['subcategory_1', 'subcategory_2'])
        df = normalize_play_age(df)
        df = encode_column(df, "age_group", AGE_MAPPING)
        df = normalize_playing_time(df)
        df = encode_column(df, "playing_time_group", PLAYING_TIME_MAPPING)
        return df
    except Exception as e:
        raise ValueError(f"Error running data preparation: {e}")
