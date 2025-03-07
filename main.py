import asyncio
from bgg.bgg import main_bgg
import data_prep as data_prep
from model.binning_input_games import bin_board_games
from model.cluster_games import bgClusters
from model.recommend_games import BoardGameRecommendation

constraint_columns = ['GAME_CAT_GROUP_card_game']

if __name__ == "__main__":
    # force_restart = True
    # df = asyncio.run(main_bgg(force_restart=force_restart))
    # df = data_prep.run_data_preparation(df)
    # df.write_json("game_data.json")

    # import polars as pl
    # df = pl.read_json("game_data.json")
    # game_names = ['Illuminati']    
    # bins = bin_board_games(df, game_names)
    # clusters = cluster_all_games(df, 5)
    # rec = BoardGameRecommendation()
    # recommendations = rec.recommend_games(clusters, bins, df)
    # 
    # # Print recommendations for each bin
    # for bin_number, data in recommendations.items():
    #     games = data['games_in_bin']
    #     recommended = data['recommended_games']
    #     
    #     # Handle singular/plural for better grammar
    #     if len(games) == 1:
    #         print(f"If you like {games[0]}, then you might enjoy:")
    #     else:
    #         game_list = ", ".join(games)
    #         print(f"If you like {game_list}, then you might enjoy:")
    #         
    #     # Print recommendations with bullet points and explanations
    #     for game in recommended:
    #         print(f"• {game}")
    #         game_scores = data["recommendation_scores"][game]
    #         explanation = rec.explain_recommendation(game_scores)
    #         print(f"  Why? {explanation}")
    #     print()
    
    import polars as pl
    df = pl.read_parquet("bgg/data/raw_bgg_data_20250216.parquet")

    df = data_prep.run_data_preparation(df)

    model = bgClusters()
    clusters = model.fit(df, constraint_columns=constraint_columns, name_column="game_name")

    rec = BoardGameRecommendation()
    game_name = 'Splendor'
    recommendations = rec.recommend_games(clusters.clusters, game_name, df)
    
    # Check if there was an error in the recommendations
    if "error" in recommendations:
        print(f"Error: {recommendations['error']}")
    else:
        # Print the input game and recommendations
        print(f"If you like {recommendations['input_game']}, then you might enjoy:")
        
        # Print recommendations with bullet points and explanations
        for game in recommendations['recommended_games']:
            print(f"• {game}")
            game_scores = recommendations["recommendation_scores"][game]
            explanation = rec.explain_recommendation(game_scores)
            print(f"  Why? {explanation}")
        print()
        
        # Print additional information if desired
        print(f"These games are from cluster {recommendations['cluster_id']}")





