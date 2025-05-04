# -*- coding: utf-8 -*-
from typing import List, TypeVar

import polars as pl


Number = TypeVar('Number', int, float)

def get_feature_columns(df: pl.DataFrame, columns: List[str]) -> List[str]:
    """
    Gets all feature columns based on configured prefixes.
    
    Args:
        df: Dataframe containing game data.
        
    Returns:
        List of column names matching the configured prefixes.
    """
    feature_columns = []
    for prefix in columns:
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        feature_columns.extend(matching_cols)

    return feature_columns

def normalize(scores: List[Number], higher_is_better: bool = True) -> List[float]:
    """
    Normalizes a list of scores to values between 0 and 1.

    Args:
        scores: List of numerical values to normalize.
        higher_is_better: Boolean indicating if higher scores are better (True) or lower scores are better (False).
            Defaults to True.

    Returns:
        list: Normalized scores between 0 and 1, where:
            - If higher_is_better=True: 1 represents the highest score
            - If higher_is_better=False: 1 represents the lowest score
            - If all scores are equal and higher_is_better=True: all values will be 1
            - If all scores are equal and higher_is_better=False: all values will be 0
    """
    if all(s == scores[0] for s in scores):
        return [1] * len(scores) if higher_is_better else [0] * len(scores)
    
    min_val, max_val = min(scores), max(scores)
    normalized = [(s - min_val) / (max_val - min_val) for s in scores]
    
    if not higher_is_better:
        normalized = [1 - n for n in normalized]
        
    return normalized