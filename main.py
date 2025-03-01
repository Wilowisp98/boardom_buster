import asyncio
from bgg.bgg import main_bgg
import data_prep as data_prep
from model.binning_input_games import bin_board_games
from model.cluster_games import cluster_all_games
from model.recommend_games import BoardGameRecommendation

if __name__ == "__main__":
    # force_restart = True
    # df = asyncio.run(main_bgg(force_restart=force_restart))
    # df = data_prep.run_data_preparation(df)
    # df.write_json("game_data.json")

    import polars as pl
    df = pl.read_json("game_data.json")
    game_names = ['Illuminati']    
    bins = bin_board_games(df, game_names)
    clusters = cluster_all_games(df, 5)
    rec = BoardGameRecommendation()
    recommendations = rec.recommend_games(clusters, bins, df)
    
    # Print recommendations for each bin
    for bin_number, data in recommendations.items():
        games = data['games_in_bin']
        recommended = data['recommended_games']
        
        # Handle singular/plural for better grammar
        if len(games) == 1:
            print(f"If you like {games[0]}, then you might enjoy:")
        else:
            game_list = ", ".join(games)
            print(f"If you like {game_list}, then you might enjoy:")
            
        # Print recommendations with bullet points and explanations
        for game in recommended:
            print(f"• {game}")
            game_scores = data["recommendation_scores"][game]
            explanation = rec.explain_recommendation(game_scores)
            print(f"  Why? {explanation}")
        print()



