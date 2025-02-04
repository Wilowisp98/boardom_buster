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