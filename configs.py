# -*- coding: utf-8 -*-
import os
from typing import List

CONSTRAINT_COLUMNS: List[str] = ['GAME_CAT_GROUP_card_game']

FEEDBACK_DIR: str = "feedback"
FEEDBACK_FILE: str = os.path.join(FEEDBACK_DIR, "user_feedback.csv")
FEEDBACK_STRUCTURE: dict = {
    'timestamp': [],
    'input_game': [],
    'recommended_game': [],
    'feedback': [],
    'reason': [],
    'comment': []
}

BGG_PATH: str = os.path.join('src', os.path.join('bgg', 'data'))
PREP_PATH: str = os.path.join('src', os.path.join('data_processing', 'processed_data'))
MODEL_PATH: str = os.path.join('src', os.path.join(os.path.join('model', 'clustering_results'), 'cluster'))

NAME_COLUMN: str = 'game_name'