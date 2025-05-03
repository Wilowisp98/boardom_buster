from typing import Dict, List

LANGUAGE_DEPENDENCY_MAPPING = {
    'No necessary in-game text': 'none',
    'Some necessary text - easily memorized or small crib sheet': 'low',
    'Moderate in-game text - needs crib sheet or paste ups': 'medium',
    'Extensive use of text - massive conversion needed to be playable': 'high',
    'Unplayable in another language': 'extreme',
    None: None
}

POPULARITY_WEIGHTS: Dict[str, float] = {
    "owned_by": 0.35,
    "wished_by": 0.25,
    "num_rates": 0.20,
    "avg_rating": 0.20
}

POPULARITY_WEIGHTS: Dict[str, float] = {
    "owned_by": 0.35,
    "wished_by": 0.25,
    "num_rates": 0.20,
    "avg_rating": 0.20
}

UNNECESSARY_COLUMNS: List[str] = [
    "best_num_players",
    "families",
    "designers",
    "artists",
    "publishers",
    "min_playtime",
    "max_playtime",
    "min_age"
]

# Maybe take something related to significance to define the minimum values.
MIN_NR_RATINGS: int = 20
MIN_RATING: float = 6.0

BOARD_GAME_CATEGORIES: Dict[str, List[str]] = {
    'time_period_historical': [
        'ancient', 'medieval', 'renaissance', 'age of reason', 'american west', 'arabian', 'napoleonic', 'post-napoleonic', 'prehistoric'
    ],
    'military_conflict': [
        'world war i', 'world war ii', 'korean war', 'vietnam war', 'modern warfare', 'american revolutionary war', 'american civil war', 'civil war', 'american indian wars', 'pike and shot', 'wargame'
    ],
    'based_on_media': [
        'movies / tv / radio theme', 'video game theme', 'book', 'novel-based', 'comic book / strip', 'music'
    ],
    'crime_espionage': [
        'murder / mystery', 'spies / secret agents', 'mafia'
    ],
    'fantasy_supernatural': [
        'fantasy', 'science fiction', 'mythology', 'zombies', 'pirates'
    ],
    'horror': [
        'horror'
    ],
    'resource_management': [
        'economic', 'industry / manufacturing', 'farming', 'city building', 'civilization', 'territory building'
    ],
    'card_game': [
        'card game'
    ],
    'physical_components': [
        'dice', 'miniatures', 'collectible components'
    ],
    'print_and_play': [
        'print & play'
    ],
    'memory': [
        'memory'
    ],
    'trivia': [
        'trivia'
    ],
    'deduction': [
        'deduction'
    ],
    'vehicles_movement': [
        'aviation / flight', 'trains', 'transportation', 'racing'
    ],
    'negotiation': [
        'negotiation', 'bluffing'
        ],
    'mental_skill': [
        'educational', 'math', 'number', 'word game', 'puzzle'
    ],
    'social_entertainment': [
        'party game', 'children\'s game', 'humor', 'mature / adult'
    ],
    'journey_discovery': [
        'adventure', 'exploration', 'travel', 'maze', 'nautical', 'space exploration'
    ],
    'real_world_topics': [
        'religious', 'political', 'environmental', 'animals'
    ],
    'sports': [
        'sports'
    ],
    'fighting': [
        'fighting'
    ],
    'abstract_strategy': [
        'abstract strategy'
    ],
    'real_time': [
        'real-time'
    ],
    'action_dexterity': [
        'action / dexterity'
    ],
    'game_system': [
        'game system'
    ],
    'electronic' : [
        'electronic'
    ]
}
