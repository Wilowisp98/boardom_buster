import asyncio
from bgg.bgg import main_bgg
import data_prep as data_prep
from model.binning_input_games import bin_board_games
from model.cluster_games import cluster_all_games
from model.recommend_games import recommend_games

if __name__ == "__main__":
    # force_restart = True
    # df = asyncio.run(main_bgg(force_restart=force_restart))
    # df = data_prep.run_data_preparation(df)
    # df.write_json("game_data.json")
    
    import polars as pl
    df = pl.read_json("game_data.json")
    game_names = df.sample(n=10, shuffle=True).select("game_name").to_series().to_list()
    bins = bin_board_games(df, game_names)
    clusters = cluster_all_games(df, 5)
    recomendations = recommend_games(clusters, bins, df)

    print(recomendations)
    



