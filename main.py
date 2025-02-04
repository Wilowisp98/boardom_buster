import asyncio
import bgg
import data_prep
import polars as pl

if __name__ == "__main__":
    game_ids = [1, 2, 3, 4, 5]
    df = asyncio.run(bgg.main(game_ids))
    
    df = data_prep.clean_text_column(df, "description")
    df = data_prep.add_popularity_score(df)
    df.write_json("game_data.json")
