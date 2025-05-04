# -*- coding: utf-8 -*-
import os

import polars as pl

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
BASE_URL: str = "https://boardgamegeek.com/xmlapi2"
CONTROL_FILE: str = os.path.join(BASE_DIR, "bgg_control.json")
DATA_DIR: str = os.path.join(BASE_DIR, "data")
BASE_FILENAME: str = "raw_bgg_data"
MAX_CHUNK_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_CONSECUTIVE_FAILURES = 50

SCHEMA = pl.Schema({
    "game_name": pl.Utf8,
    "description": pl.Utf8,
    "subcategory_1": pl.Utf8,
    "subcategory_2": pl.Utf8,
    "publication_year": pl.Int32,
    "min_players": pl.Int32,
    "max_players": pl.Int32,
    "best_num_players": pl.Utf8,
    "recommended_num_players": pl.Utf8,
    "suggested_play_age": pl.Int32,
    "categories": pl.List(pl.Utf8),
    "mechanics": pl.List(pl.Utf8),
    "families": pl.List(pl.Utf8),
    "designers": pl.List(pl.Utf8),
    "artists": pl.List(pl.Utf8),
    "publishers": pl.List(pl.Utf8),
    "playing_time": pl.Int32,
    "min_playtime": pl.Int32,
    "max_playtime": pl.Int32,
    "min_age": pl.Int32,
    "language_dependence_description": pl.Utf8,
    "game_rank": pl.Int32,
    "avg_rating": pl.Float64,
    "num_rates": pl.Int32,
    "rank_subcategory_1": pl.Int32,
    "rank_subcategory_2": pl.Int32,
    "avg_weight": pl.Float64,
    "num_weights": pl.Int32,
    "owned_by": pl.Int32,
    "wished_by": pl.Int32
})
