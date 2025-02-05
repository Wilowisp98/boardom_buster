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

def one_hot_encode(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Perform one-hot encoding on a specified column of a Polars DataFrame.
    
    Args:
        df (pl.DataFrame): Input DataFrame
        column (str): The column to one-hot encode
    
    Returns:
        pl.DataFrame: Original DataFrame with one-hot encoded columns added
    
    Example:
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B", "Game C"],
        ...     "category": ["Strategy", "Party", "Strategy"]
        ... })
        >>> one_hot_encode(df, "category")
        shape: (3, 4)
        ┌───────────┬───────────┬──────────┬────────┐
        │ game_name ┆ category  ┆ Party    ┆ Strategy │
        │ ---       ┆ ---       ┆ ---      ┆ ---      │
        │ str       ┆ str       ┆ u8       ┆ u8       │
        ╞═══════════╪═══════════╪══════════╪══════════╡
        │ Game A    ┆ Strategy  ┆ 0        ┆ 1        │
        │ Game B    ┆ Party     ┆ 1        ┆ 0        │
        │ Game C    ┆ Strategy  ┆ 0        ┆ 1        │
        └───────────┴───────────┴──────────┴──────────┘
    """
    unique_values = df[column].unique().to_series().to_list()
    
    encoded_cols = [
        (pl.col(column) == value).cast(pl.UInt8).alias(value) 
        for value in unique_values
    ]
    
    return df.with_columns(encoded_cols)

def one_hot_encode_multiple(df: pl.DataFrame, columns: list) -> pl.DataFrame:
    """
    Perform one-hot encoding on multiple categorical columns by creating binary indicator variables 
    for each unique category across the specified columns.

    This function ensures that if a category appears in either of the specified columns, 
    it will be counted as present in the one-hot encoded output.

    Args:
        df (pl.DataFrame): Input DataFrame containing categorical columns.
        columns (list): List of categorical column names to be one-hot encoded.

    Returns:
        pl.DataFrame: Original DataFrame with additional binary columns for one-hot encoding.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "game_name": ["Game A", "Game B", "Game C"],
        ...     "subcategory_1": ["Strategy", "Party", "Strategy"],
        ...     "subcategory_2": ["Economic", "Strategy", None]
        ... })
        >>> df_encoded = one_hot_encode_multiple(df, ["subcategory_1", "subcategory_2"])
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
    
    Notes:
        - This function handles missing values (`None`) automatically.
        - Uses `pl.any_horizontal()` to check for a category across multiple columns.
        - Converts results to `UInt8` (binary representation).
    """
    # Get unique categories from both columns
    unique_values = (
        df.select(columns)
        .melt()
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )
    
    # Generate one-hot encoding for each unique value
    encoded_cols = [
        (pl.any_horizontal([pl.col(col) == value for col in columns]))
        .cast(pl.UInt8)
        .alias(value)
        for value in unique_values
    ]
    
    return df.with_columns(encoded_cols)

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
        │ game_name │ suggested_play_age │ age_group │
        │ ---       │ ---                │ ---       │
        │ str       │ i64                │ str       │
        ╞═══════════╪════════════════════╪═══════════╡
        │ Game A    │ 6                  │ Kids      │
        │ Game B    │ 10                 │ Family    │
        │ Game C    │ 14                 │ Teen      │
        │ Game D    │ 18                 │ Adult     │
        └───────────┴────────────────────┴───────────┘
    """
    age_group_col = (
        pl.when(pl.col("suggested_play_age") <= 8).then("Kids")
        .when(pl.col("suggested_play_age") <= 12).then("Family")
        .when(pl.col("suggested_play_age") <= 16).then("Teen")
        .otherwise("Adult")
        .alias("age_group")
    )
    
    return df.with_columns(age_group_col)

def encode_column(df: pl.DataFrame, column_name: str, mapping_dict: dict) -> pl.DataFrame:
    """
    General function to encode a categorical column using a provided mapping dictionary.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column_name (str): Name of the column to encode.
        mapping_dict (dict): Dictionary for mapping categories to numerical values.

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
    encoded_col = pl.col(column_name).map_dict(mapping_dict).alias(f"{column_name}_encoded")
    
    return df.with_columns(encoded_col)

def normalize_playing_time(df: pl.DataFrame) -> pl.DataFrame:
    """
    Categorize the playing time into buckets based on the duration.
    
    Args:
        df (pl.DataFrame): Input DataFrame.
        column_name (str): Name of the playing time column.
        
    Returns:
        pl.DataFrame: Updated DataFrame with categorized playing time.
    
    Example:
        >>> df = pl.DataFrame({"playing_time": [15, 45, 75, 140]})
        >>> categorize_playing_time(df, "playing_time")
        shape: (4, 2)
        ┌──────────────┬────────────────────────┐
        │ playing_time │ playing_time_group_col │
        │ ---          │ ---                    │
        │ i64          │ str                    │
        ╞══════════════╪════════════════════════╡
        │ 15           │ Short                  │
        │ 45           │ Medium                 │
        │ 75           │ Long                   │
        │ 140          │ Very Long              │
        └──────────────┴────────────────────────┘
    """
    playing_time_group_col = (
        pl.when(pl.col("playing_time") <= 30).then("Short")
        .when(pl.col("playing_time") <= 60).then("Medium")
        .when(pl.col("playing_time") <= 12).then("Long")
        .otherwise("Very Long")
        .alias("playing_time_group")
    )
    
    return df.with_columns(playing_time_group_col)

def run_data_preparation(df: pl.DataFrame) -> pl.DataFrame:

    df = clean_text_column(df, "description")
    df = add_popularity_score(df)
    df = one_hot_encode_multiple(df, ['subcategory_1', 'subcategory_2'])
    # df = normalize_play_age(df)
    # age_mapping = {"Kids": 0, "Family": 1, "Teen": 2, "Adult": 3}
    # df = encode_column(df, "age_group", age_mapping)
    # df = normalize_playing_time(df)
    # playing_mapping = {"Short": 0, "Medium": 1, "Long": 2, "Very Long": 3}
    # df = encode_column(df, "playing_time_group", playing_mapping)

    return df