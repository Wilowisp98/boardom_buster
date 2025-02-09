import asyncio
from bgg.bgg import main_bgg
import data_prep as data_prep
from model.binning_games import bin_board_games

if __name__ == "__main__":
    force_restart = True
    df = asyncio.run(main_bgg(force_restart=force_restart))
    df = data_prep.run_data_preparation(df)
    df.write_json("game_data.json")
    
    game_names = df.sample(n=10, shuffle=True).select("game_name").to_series().to_list()
    bins = bin_board_games(df, game_names)
    print(bins)



