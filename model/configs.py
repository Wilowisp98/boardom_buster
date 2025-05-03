from typing import List

PLOT = False

RELEVANT_COLUMNS: List[str] = [
    "AGE_GROUP",
    "GAME_CAT",
    "LANGUAGE_DEPENDENCY",
    "GAME_DURATION",
    "GAME_DIFFICULTY"
]

FINAL_MODEL_DIR: str = "model/clustering_results/cluster"
MODEL_FILENAME: str = "clusters.pkl"