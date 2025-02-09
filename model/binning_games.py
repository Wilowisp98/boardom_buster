import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Tuple

RELEVANT_COLUMNS: List[str] = [
    "OHE_familygames",  # Game type
    "OHE_thematic",
    "OHE_childrensgames",
    "OHE_wargames",
    "OHE_abstracts",
    "OHE_cgs",
    "OHE_strategygames",
    "OHE_partygames",
    "is_solo_game",  # Game players
    "is_couple_game",
    "is_small_group_game",
    "is_party_game",
    "OHE_Kids",  # Game Age
    "OHE_Teen",
    "OHE_Family",
    "OHE_Adult",
    "OHE_Unknown",  # Game length
    "OHE_Short",
    "OHE_Medium",
    "OHE_Long",
    "OHE_Very Long"
]

def bin_board_games(df: pl.DataFrame, input_board_games: List[str], threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Bins the input board games based on cosine similarity.

    Args:
        df (pl.DataFrame): The DataFrame containing all board games.
        input_board_games (List[str]): List of board game names to bin.
        threshold (float): The similarity threshold for binning.

    Returns:
        List[Dict[str, Any]]: A list of bins, each containing members and a centroid.
    """
    # Filter the DataFrame to include only the input board games
    filtered_df = df.filter(pl.col("game_name").is_in(input_board_games))
    filtered_columns_df = filtered_df.select(['game_name'] + RELEVANT_COLUMNS)
    
    # Create a mapping between indices and game names
    game_names = filtered_columns_df.select('game_name').to_series().to_list()
    
    # Convert the DataFrame to a NumPy array for similarity computation (excluding game_name)
    data_array = filtered_columns_df.select(RELEVANT_COLUMNS).to_numpy()

    print(data_array)
    bins = []

    # Iterate over each board game
    for idx, row_values in enumerate(data_array):
        game_name = game_names[idx]
        
        if not bins:
            bins.append({
                'centroid': row_values,
                'members': [(game_name, row_values)]
            })
            continue

        similar_bin = None
        similarity = -np.inf
        for bin in bins:
            current_similarity = cosine_similarity([row_values], [bin['centroid']])[0][0]
            if current_similarity > max(similarity, threshold):
                similar_bin = bin
                similarity = current_similarity

        if similar_bin:
            similar_bin['members'].append((game_name, row_values))
            # Extract just the feature vectors for centroid calculation
            members_array = np.array([m[1] for m in similar_bin['members']])
            new_centroid = np.mean(members_array, axis=0)

            # Round one-hot encoded features to integers
            new_centroid = np.round(new_centroid).astype(int)
            
            similar_bin['centroid'] = new_centroid
        else:
            bins.append({
                'members': [(game_name, row_values)], 
                'centroid': row_values
            })

    # Format the output to be more readable
    formatted_bins = []
    for idx, bin in enumerate(bins):
        formatted_bin = {
            'bin_id': idx + 1,
            'games': [member[0] for member in bin['members']],  # Just the game names
            'size': len(bin['members']),
            'centroid': bin['centroid'].tolist()
        }
        formatted_bins.append(formatted_bin)

    return formatted_bins