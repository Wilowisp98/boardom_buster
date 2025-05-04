# -*- coding: utf-8 -*-
from typing import List
import os

PLOT = False

RELEVANT_COLUMNS: List[str] = [
    "AGE_GROUP",
    "GAME_CAT",
    "LANGUAGE_DEPENDENCY",
    "GAME_DURATION",
    "GAME_DIFFICULTY"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DETAILS_DIR: str = os.path.join(BASE_DIR, "clustering_results")
FINAL_MODEL_DIR: str = os.path.join(os.path.join(BASE_DIR, "clustering_results"), "cluster")
MODEL_FILENAME: str = "clusters.pkl"

POPULARITY_WEIGHT: float = 0.25
DISTANCE_WEIGHT: float = 0.55
RATING_QUALITY_WEIGHT: float = 0.20

MIN_SIMILARITY_THRESHOLD: float = 0.4