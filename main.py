import asyncio
from bgg.bgg import main_bgg
import data_prep

if __name__ == "__main__":
    df = asyncio.run(main_bgg(force_restart=True))
    # df = data_prep.run_data_preparation(df)
    # df.write_json("game_data.json")
