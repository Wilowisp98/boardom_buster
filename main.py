import asyncio
from bgg.bgg import main_bgg
import data_prep
import polars as pl

if __name__ == "__main__":
    game_ids = [1, 2, 3, 4, 5]
    df = asyncio.run(main_bgg(force_restart=True))
    df = data_prep.run_data_preparation(df)
    df.write_json("game_data.json")
