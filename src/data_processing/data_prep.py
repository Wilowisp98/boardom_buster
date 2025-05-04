# -*- coding: utf-8 -*-
import html
import os
import re
from datetime import datetime
from typing import Dict, List

import polars as pl

from .configs import *

# TO DO:
# - Study more about scores.

def group_categories(df: pl.DataFrame) -> pl.DataFrame:
    """
    Group individual board game categories into broader category groups based on the BOARD_GAME_CATEGORIES 
    configuration. Creates binary indicator columns for each category group and removes the original 
    individual category columns.

    Args:
        df (pl.DataFrame): DataFrame containing individual board game category columns 
            (named as GAME_CAT_{category}).

    Returns:
        pl.DataFrame: DataFrame with new category group columns (GAME_CAT_GROUP_{group_name}) 
            and original individual category columns removed.

    Examples:
        >>> import polars as pl
        >>> from configs import BOARD_GAME_CATEGORIES
        >>> # Assuming BOARD_GAME_CATEGORIES = {"Strategy": ["Abstract", "AreaControl"]}
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B", "Game C"],
        ...     "GAME_CAT_Abstract": [1, 0, 1],
        ...     "GAME_CAT_AreaControl": [0, 1, 1]
        ... })
        >>> group_categories(df)
        shape: (3, 2)
        ┌───────────┬─────────────────────────┐
        │ game_name ┆ GAME_CAT_GROUP_Strategy │
        │ ---       ┆ ---                     │
        │ str       ┆ i32                     │
        ╞═══════════╪═════════════════════════╡
        │ Game A    ┆ 1                       │
        │ Game B    ┆ 1                       │
        │ Game C    ┆ 1                       │
        └───────────┴─────────────────────────┘
    """
    try:
        for group_name in BOARD_GAME_CATEGORIES.keys():
            group_col_name = f"GAME_CAT_GROUP_{group_name}"
            categories_in_group = BOARD_GAME_CATEGORIES[group_name]

            group_expr = pl.lit(False)
            for category in categories_in_group:
                category_col = f"GAME_CAT_{category}"
                if category_col in df.columns:
                    group_expr = group_expr | (pl.col(category_col) == 1)

            category_column = (
                pl.when(group_expr)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
            ).alias(group_col_name)
            df = df.with_columns(category_column)

        all_categories = []
        for categories_list in BOARD_GAME_CATEGORIES.values():
            all_categories.extend(categories_list)

        columns_to_drop = [f"GAME_CAT_{category}" for category in all_categories if f"GAME_CAT_{category}" in df.columns]
        df = df.drop(columns_to_drop)

        return df
    except Exception as e:
        raise ValueError(f"Error while with group category {group_name}: {e}")

def add_popularity_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a popularity score column to the games DataFrame based on ownership, wishlists, ratings count,
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

        df = df.with_columns(popularity_col)

        return df
    except Exception as e:
        raise ValueError(f"Error adding popularity score: {e}")
    
def encode_column(df: pl.DataFrame, column_name: str, mapping_dict: Dict[str, int]) -> pl.DataFrame:
    """
    General function to encode a categorical column using a provided mapping dictionary.
    Rows with null values in the specified column are excluded.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column_name (str): Name of the column to encode.
        mapping_dict (Dict[str, int]): Dictionary for mapping categories to numerical values.

    Returns:
        pl.DataFrame: Updated DataFrame with the encoded column.

    Example:
        >>> df = pl.DataFrame({"age_group": ["Kids", "Family", "Teen", "Adult", None]})
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
        df_filtered = df.filter(pl.col(column_name).is_not_null())
        encoded_col = pl.col(column_name).replace(mapping_dict)
        
        all_integers = all(isinstance(v, int) for v in mapping_dict.values())
        if all_integers:
            encoded_col = encoded_col.cast(pl.Int32)
        
        df = df_filtered.with_columns(encoded_col.alias(f"{column_name}_encoded"))

        return df
    except Exception as e:
        raise ValueError(f"Error encoding column '{column_name}': {e}")

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

        df = df.with_columns(cleaned_col)

        return df
    except Exception as e:
        raise ValueError(f"Error cleaning {column}: {e}")

def one_hot_encode(df: pl.DataFrame, columns: List[str], prefix: str = None) -> pl.DataFrame:
    """
    Perform one-hot encoding on one or multiple categorical columns that may contain either single values
    or lists of values. Creates binary indicator variables for each unique category across all specified columns.
    Rows with no values in the specified columns are removed.

    Args:
        df (pl.DataFrame): Input DataFrame containing categorical columns.
        columns (List[str]): List of categorical column names to be one-hot encoded.
        prefix (str, optional): Prefix to add to the generated column names.

    Returns:
        pl.DataFrame: Original DataFrame with additional binary columns for one-hot encoding.
    """
    try:
        all_unique_values = set()
        for column in columns:
            sample_value = df.select(pl.col(column)).row(0)[0]
            is_list_column = isinstance(sample_value, list)
            
            if is_list_column:
                unique_values = (
                    df.select(pl.col(column))
                    .filter(pl.col(column).is_not_null())
                    .explode(column)
                    .unique()
                    .to_series()
                    .to_list()
                )
            else:
                unique_values = (
                    df.select(pl.col(column))
                    .filter(pl.col(column).is_not_null())
                    .unique()
                    .to_series()
                    .to_list()
                )

            all_unique_values.update(unique_values)

        for value in all_unique_values:
            binary_exprs = []
            for col in columns:
                sample_value = df.select(pl.col(col)).row(0)[0]
                is_list_column = isinstance(sample_value, list)

                if is_list_column:
                    binary_exprs.append(
                        pl.col(col).map_elements(
                            lambda x: 1 if (x is not None and value in x) else 0,
                            return_dtype=pl.UInt8
                        )
                    )
                else:
                    binary_exprs.append(
                        (pl.col(col).fill_null(pl.lit("__NULL__")) == value).cast(pl.UInt8)
                    )
            
            column_name = f"{prefix}_{str(value).lower()}" if prefix else f"{str(value).lower()}"
            df = df.with_columns(
                pl.max_horizontal(binary_exprs).alias(column_name)
            )

        return df
    except Exception as e:
        raise ValueError(f"Error performing one-hot encoding: {str(e)}")

def normalize_play_age(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the suggested play age into broader categories: Kids, Family, Teen, and Adult.
    Exclude rows with "Unknown" age groups.

    Args:
        df (pl.DataFrame): Input DataFrame with the 'suggested_play_age' column.

    Returns:
        pl.DataFrame: Updated DataFrame with an additional 'age_group' column, excluding "Unknown" rows.

    Example:
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B", "Game C", "Game D", "Game E"],
        ...     "suggested_play_age": [6, 10, 14, 18, None]
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

        age_grouped_df = df.with_columns(age_group_col)
        age_grouped_df = age_grouped_df.filter(pl.col("age_group") != "Unknown")

        return age_grouped_df
    except Exception as e:
        raise ValueError(f"Error normalizing play age: {e}")

def normalize_playing_time(df: pl.DataFrame) -> pl.DataFrame:
    """
    Categorize the playing time into buckets based on the duration and exclude rows with "Unknown" playing time.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: Updated DataFrame with categorized playing time, excluding "Unknown" rows.

    Example:
        >>> df = pl.DataFrame({"playing_time": [15, 45, 75, 140, None, 2000]})
        >>> normalize_playing_time(df)
        shape: (4, 2)
        ┌──────────────┬────────────────────────┐
        │ playing_time ┆ playing_time_group_col │
        │ ---          ┆ ---                    │
        │ i64          ┆ str                    │
        ╞══════════════╪════════════════════════╡
        │ 15           ┆ Short                  │
        │ 45           ┆ Medium                 │
        │ 75           ┆ Long                   │
        │ 140          ┆ Very_Long              │
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

        df_with_group = df.with_columns(playing_time_group_col)
        df_filtered = df_with_group.filter(pl.col("playing_time_group") != "Unknown")

        return df_filtered
    except Exception as e:
        raise ValueError(f"Error categorizing playing time: {e}")

# I'm not normalizing and then one-hot encoding the same way as the I'm doing on other columns because a 
# game can fill into multiple categories regarding player count.
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
        df = df.with_columns([
            (pl.col("min_players") <= 1).cast(pl.UInt8).alias("GROUP_SIZE_solo"),
            ((pl.col("min_players") <= 2) & (pl.col("max_players") >= 2)).cast(pl.UInt8).alias("GROUP_SIZE_couple"),
            ((pl.col("min_players") >= 3) & (pl.col("max_players") <= 5)).cast(pl.UInt8).alias("GROUP_SIZE_small"),
            (pl.col("max_players") >= 6).cast(pl.UInt8).alias("GROUP_SIZE_party")
        ])
        
        return df
    except Exception as e:
        raise ValueError(f"Error normalizing player count: {str(e)}")

def normalize_game_difficulty(df: pl.DataFrame) -> pl.DataFrame:
    """
    Categorize the game difficulty into buckets based on the "avg_weight" and exclude rows with "Unknown" playing time.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: Updated DataFrame with categorized playing time, excluding "Unknown" rows.

    Example:
        >>> df = pl.DataFrame({"avg_weight": [1, 2, 3, 4]})
        >>> normalize_game_difficulty(df)
        shape: (4, 2)
        ┌──────────────┬────────────────────────┐
        │ avg_weight   ┆ game_difficulty_group  │
        │ ---          ┆ ---                    │
        │ i64          ┆ str                    │
        ╞══════════════╪════════════════════════╡
        │ 1            ┆ Easy                   │
        │ 2            ┆ Medium                 │
        │ 3            ┆ Hard                   │
        │ 4            ┆ Very_Hard              │
        └──────────────┴────────────────────────┘
    """
    try:
        game_difficulty_col = (
            pl.when(pl.col("avg_weight") <= 1).then(pl.lit("Very Easy"))
            .when((pl.col("avg_weight") > 1) & (pl.col("avg_weight") <= 2)).then(pl.lit("Easy"))
            .when((pl.col("avg_weight") > 2) & (pl.col("avg_weight") <= 3)).then(pl.lit("Medium"))
            .when((pl.col("avg_weight") > 3) & (pl.col("avg_weight") <= 4)).then(pl.lit("Hard"))
            .when((pl.col("avg_weight") > 4) & (pl.col("avg_weight") <= 5)).then(pl.lit("Very_Hard"))
            .otherwise(pl.lit("Unknown"))
            .alias("game_difficulty_group")
        )

        df_with_group = df.with_columns(game_difficulty_col)
        df_filtered = df_with_group.filter(pl.col("game_difficulty_group") != "Unknown")

        return df_filtered
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
        df = (df.filter(pl.col("num_rates") >= MIN_NR_RATINGS)
                .filter(pl.col("avg_rating") >= MIN_RATING))
        df = clean_text_column(df, "description")
        df = add_popularity_score(df)
        df = normalize_play_age(df)
        df = one_hot_encode(df, ['age_group'], 'AGE_GROUP')
        df = normalize_playing_time(df)
        df = one_hot_encode(df, ['playing_time_group'], 'GAME_DURATION')
        df = normalize_game_difficulty(df)
        df = one_hot_encode(df, ['game_difficulty_group'], 'GAME_DIFFICULTY')
        df = normalize_player_count(df)
        df = one_hot_encode(df, ['categories'], 'GAME_CAT')
        df = group_categories(df)
        df = encode_column(df, 'language_dependence_description', LANGUAGE_DEPENDENCY_MAPPING)
        df = one_hot_encode(df, ['language_dependence_description_encoded'], 'LANGUAGE_DEPENDENCY')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "processed_data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"game_data_{int(datetime.now().strftime('%Y%m%d'))}.parquet")
        
        df.write_parquet(output_path)
        return df
    except Exception as e:
        raise ValueError(f"Error running data preparation: {e}")