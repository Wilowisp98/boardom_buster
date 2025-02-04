import polars as pl
import html
import re

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
    cleaned_col = (
        pl.col(column)
        .map_elements(lambda x: 
            re.sub(r'\s+', ' ',             # Replace all whitespace (including \n) with single space
                html.unescape(str(x))       # Convert HTML entities
                .replace('&', '')           # Remove remaining &
                .replace('-', '')           # Remove -
                .replace('\\', '')          # Remove backslashes
                .replace('"', '')           # Remove quotes
            ).strip()                       # Remove leading/trailing whitespace
            if x is not None else None,
            return_dtype=pl.String
        )
        .alias(column)
    )
    
    return df.with_columns(cleaned_col)

def add_popularity_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a popularity score column to a games DataFrame based on ownership, wishlists, ratings count, 
    and average rating. The score is normalized between 0 and 1.

    Args:
        df (pl.DataFrame): DataFrame containing game metrics columns: owned_by, wished_by, 
            num_rates, and avg_rating

    Returns:
        pl.DataFrame: Original DataFrame with an additional 'popularity_score' column

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
    popularity_col = (
        (pl.col("owned_by") / pl.col("owned_by").max() * 0.4) +
        (pl.col("wished_by") / pl.col("wished_by").max() * 0.2) +
        (pl.col("num_rates") / pl.col("num_rates").max() * 0.2) +
        (pl.col("avg_rating") / 10 * 0.2)
    ).alias("popularity_score")
    
    return df.with_columns(popularity_col)