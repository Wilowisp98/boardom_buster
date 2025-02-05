import asyncio
import bgg
import data_prep
import polars as pl

if __name__ == "__main__":
    game_ids = [1, 2, 3, 4, 5]
    df = asyncio.run(bgg.main(game_ids))
    
    df = data_prep.run_data_preparation(df)
    df.write_json("game_data.json")
