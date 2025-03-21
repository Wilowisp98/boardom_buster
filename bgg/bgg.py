import os
import json
import time
import asyncio
import httpx
import xmltodict
import polars as pl
from datetime import datetime
from typing import List, Optional, Tuple
from logger import get_logger
from .schema import SCHEMA

class BGGConfig:
    """
    BGGClient Configuration class.
    """
    def __init__(self, base_url: str = "https://boardgamegeek.com/xmlapi2"):
        self.base_url = base_url
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.control_file = os.path.join(self.base_dir, "bgg_control.json")
        self.data_dir = os.path.join(self.base_dir, "data")
        self.base_filename = "raw_bgg_data"
        self.max_chunk_size = 10
        self.max_retries = 3
        self.retry_delay = 5
        self.max_consecutive_failures = 50

class BGG:
    """
    Class for interacting with the BoardGameGeek (BGG) XML API2.

    Attributes:
        config (BGGConfig): Configuration settings for the BGG client.
    """
    def __init__(self) -> None:
        self.config = BGGConfig
        self.logger = get_logger('BGGLogger', base_dir=self.config.base_dir)
        self.control_data = self._load_control_data()
        self.global_df = None
        self.current_date = int(datetime.now().strftime('%Y%m%d'))

    def _load_control_data(self) -> dict:
        """
        Loads control data from a JSON file or initializes default values if no file exists.

        Returns:
            dict: A dictionary containing control data with the following keys:
                - first_execution (bool): True if this is the first time running (no existing file)
                - last_id (int): The ID of the last processed item, defaults to 1 for new executions
        """
        if os.path.exists(self.config.control_file):
            with open(self.config.control_file, 'r') as f:
                data = json.load(f)
                self.logger.info(f"Loaded control data: last_id={data.get('last_id')}")
                return data

        self.logger.info("No existing control data found, starting fresh")
        return {"first_execution": True, "last_id": 1}

    def _save_control_data(self):
        """
        Saves the current control data to a JSON file.

        Raises:
            IOError: If there are issues writing to the control file
            TypeError: If self.control_data contains values that cannot be serialized to JSON
        """
        with open(self.config.control_file, 'w') as f:
            json.dump(self.control_data, f)

    async def get_games_data(self, game_ids: List[int]) -> List[pl.DataFrame]:
        """
        Fetches detailed information for multiple games in a single API request.

        Args:
            game_ids (List[int]): List of BGG game IDs to fetch.

        Returns:
            List[pl.DataFrame]: List of DataFrames containing the processed game data.
        """
        endpoint = f"{self.config.base_url}/thing"
        params = {
            "id": ",".join(map(str, game_ids)),
            "type": "boardgame",
            "stats": 1
        }

        async with httpx.AsyncClient() as client:
            for attempt in range(self.config.max_retries):
                try:
                    response = await client.get(endpoint, params=params)

                    if response.status_code == 200:
                        response_data = xmltodict.parse(response.content)
                        if 'items' not in response_data or not response_data['items'].get('item'):
                            self.logger.warning(f"No data found for game IDs {game_ids[0]} to {game_ids[-1]}")
                            return []

                        # Handle both single and multiple items
                        items = response_data['items']['item']
                        if not isinstance(items, list):
                            items = [items]

                        self.logger.debug(f"Successfully retrieved data for {len(items)} games")

                        await asyncio.sleep(3)
                        return [self._prepare_data({'items': {'item': item}}) for item in items if item]

                    elif response.status_code == 429:
                        self.logger.info(f"Request queued for game IDs {game_ids[0]} to {game_ids[-1]}, attempt {attempt + 1}/{self.config.max_retries}")
                        await asyncio.sleep(self.config.retry_delay * (self.config.max_retries - attempt))
                    else:
                        self.logger.error(f"HTTP {response.status_code} for game IDs {game_ids[0]} to {game_ids[-1]}")
                        response.raise_for_status()

                except httpx.HTTPStatusError as e:
                    self.logger.error(f"HTTP error retrieving game IDs {game_ids[0]} to {game_ids[-1]}: {str(e)}", exc_info=True)
                except Exception as e:
                    self.logger.error(f"Error retrieving game IDs {game_ids[0]} to {game_ids[-1]}: {str(e)}", exc_info=True)
                    if attempt == self.config.max_retries - 1:
                        raise

        raise Exception(f"Failed to get response for game IDs {game_ids[0]} to {game_ids[-1]} after {self.config.max_retries} attempts")

    def _prepare_data(self, response_data: dict) -> pl.DataFrame:
        """
        Processes the raw API response data into a structured DataFrame.

        Args:
            response_data (dict): The raw data from the API response.

        Returns:
            pl.DataFrame: A DataFrame containing the processed game data.

        Raises:
            Exception: If there is an error processing the data.
        """
        try:
            game_info = response_data['items']['item']

            # Extract basic game information
            game_name = game_info['name'][0]['@value'] if isinstance(game_info['name'], list) else game_info['name']['@value']
            game_description = game_info['description']
            game_publication_year = int(game_info['yearpublished']['@value'])
            game_min_players = int(game_info['minplayers']['@value'])
            game_max_players = int(game_info['maxplayers']['@value'])

            # Extract poll data
            best_numplayers, recommended_numplayers, suggested_playerage, language_dependence = self._extract_poll_data(game_info)

            # Extract categories, mechanics, families, designers, artists, and publishers
            game_categories = self._extract_links(game_info, 'boardgamecategory')
            game_mechanics = self._extract_links(game_info, 'boardgamemechanic')
            game_families = self._extract_links(game_info, 'boardgamefamily')
            game_designers = self._extract_links(game_info, 'boardgamedesigner')
            game_artists = self._extract_links(game_info, 'boardgameartist')
            game_publishers = self._extract_links(game_info, 'boardgamepublisher')

            # Extract playing time and age information
            game_playing_time = int(game_info['playingtime']['@value'])
            game_min_playtime = int(game_info['minplaytime']['@value'])
            game_max_playtime = int(game_info['maxplaytime']['@value'])
            game_min_age = int(game_info['minage']['@value'])

            # Inside _prepare_data method
            game_stats = game_info['statistics']
            ratings = game_stats['ratings']

            # Basic stats
            num_rates = int(ratings['usersrated']['@value'])
            avg_rating = float(ratings['average']['@value'])

            # Extract ranks and categories
            game_rank = None
            game_subcategories = []
            for rank in ratings.get('ranks', {}).get('rank', []):
                if isinstance(rank, dict):
                    if rank['@name'] == 'boardgame':
                        game_rank = int(rank['@value']) if rank['@value'] != 'Not Ranked' else None
                    elif rank['@type'] == 'family':
                        game_subcategories.append({
                            'name': rank['@name'],
                            'rank': int(rank['@value']) if rank['@value'] != 'Not Ranked' else None
                        })

            subcategory_1 = game_subcategories[0]['name'] if len(game_subcategories) > 0 else None
            rank_subcategory_1 = game_subcategories[0]['rank'] if len(game_subcategories) > 0 else None
            subcategory_2 = game_subcategories[1]['name'] if len(game_subcategories) > 1 else None
            rank_subcategory_2 = game_subcategories[1]['rank'] if len(game_subcategories) > 1 else None

            # Additional stats with type conversion
            num_weights = int(ratings['numweights']['@value'])
            avg_weight = float(ratings['averageweight']['@value'])
            owned_by = int(ratings['owned']['@value'])
            wished_by = int(ratings['wishing']['@value'])

            # Create DataFrame
            df = pl.DataFrame({
                "game_name": [game_name],
                "description": [game_description],
                "subcategory_1": [subcategory_1],
                "subcategory_2": [subcategory_2],
                "publication_year": [game_publication_year],
                "min_players": [game_min_players],
                "max_players": [game_max_players],
                "best_num_players": [best_numplayers],
                "recommended_num_players": [recommended_numplayers],
                "suggested_play_age": [suggested_playerage],
                "categories": [game_categories],
                "mechanics": [game_mechanics],
                "families": [game_families],
                "designers": [game_designers],
                "artists": [game_artists],
                "publishers": [game_publishers],
                "playing_time": [game_playing_time],
                "min_playtime": [game_min_playtime],
                "max_playtime": [game_max_playtime],
                "min_age": [game_min_age],
                "language_dependence_description": [language_dependence],
                "game_rank": [game_rank],
                "avg_rating": [avg_rating],
                "num_rates": [num_rates],
                "rank_subcategory_1": [rank_subcategory_1],
                "rank_subcategory_2": [rank_subcategory_2],
                "avg_weight": [avg_weight],
                "num_weights": [num_weights],
                "owned_by": [owned_by],
                "wished_by": [wished_by]},
                schema=SCHEMA
            )

            return df

        except KeyError as e:
            raise Exception(f"Failed to process game data: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error processing game data: {str(e)}")

    def _extract_poll_data(self, game_info: dict) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[dict]]:
        """
        Extracts poll data from the game information.

        Args:
            game_info (dict): The game information dictionary.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[int], Optional[dict]]: A tuple containing the best number of players,
            recommended number of players, suggested player age, and language dependence information.
        """
        best_numplayers = None
        recommended_numplayers = None
        suggested_playerage = None
        language_dependence = None

        for poll in game_info.get('poll', []):
            if poll['@name'] == 'suggested_numplayers':
                best_votes = 0
                recommended_votes = 0
                for result in poll['results']:
                    if isinstance(result, dict):
                        for player_result in result.get('result', []):
                            if player_result['@value'] == 'Best' and int(player_result['@numvotes']) > best_votes:
                                best_numplayers = result['@numplayers']
                                best_votes = int(player_result['@numvotes'])

                            if player_result['@value'] == 'Recommended' and int(player_result['@numvotes']) > recommended_votes:
                                recommended_numplayers = result['@numplayers']
                                recommended_votes = int(player_result['@numvotes'])

            elif poll['@name'] == 'suggested_playerage':
                max_votes = 0
                for result in poll['results'].get('result', []):
                    if isinstance(result, dict):
                        votes = int(result['@numvotes'])
                        if votes > max_votes:
                            if result['@value'] == '21 and up':
                                suggested_playerage = 21
                            else:
                                suggested_playerage = int(result['@value'])
                            max_votes = votes

            elif poll['@name'] == 'language_dependence':
                max_votes = 0
                for result in poll['results'].get('result', []):
                    if isinstance(result, dict):
                        votes = int(result['@numvotes'])
                        if votes > max_votes:
                            language_dependence = result['@value']
                            max_votes = votes

        return best_numplayers, recommended_numplayers, suggested_playerage, language_dependence


    def _extract_links(self, game_info: dict, link_type: str) -> List[str]:
        """
        Extracts specific links from the game information.

        Args:
            game_info (dict): The game information dictionary.
            link_type (str): The type of link to extract (e.g., 'boardgamecategory').

        Returns:
            List[str]: A list of link values.
        """
        return [link['@value'] for link in game_info.get('link', []) if link['@type'] == link_type]

    async def continuous_scan(self, force_restart: bool = False, batch_size: int = 30) -> None:
        """
        Continuously scans and retrieves game data in batches.

        Args:
            force_restart (bool): If True, resets the control data to start scanning from ID 1.
            batch_size (int): Number of games to process in each API request. Defaults to 100.
        """
        if force_restart:
            self.logger.info("Forcing restart of scanning process")
            self.control_data = {"first_execution": True, "last_id": 1}

        start_id = 1 if self.control_data["first_execution"] else self.control_data["last_id"]
        current_id = start_id
        dataframes = []
        consecutive_failures = 0

        self.logger.info(f"Starting continuous scan from ID {start_id}")

        # while consecutive_failures < self.config.max_consecutive_failures:
        while current_id <= 10000:
            try:
                batch_ids = list(range(current_id, current_id + batch_size))
                self.logger.info(f"Processing batch: IDs {current_id} to {current_id + batch_size - 1}")

                chunks_ids = [batch_ids[i:i+self.config.max_chunk_size] for i in range(0, len(batch_ids), self.config.max_chunk_size)]
                tasks = [self.get_games_data(chunk_ids) for chunk_ids in chunks_ids]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                valid_results = []
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Chunk failed: {str(result)}")
                    elif result:
                        valid_results.extend(result)

                if valid_results:
                    dataframes.extend(valid_results)
                    self.logger.info(f"Successfully processed {len(valid_results)} games in batch {current_id}-{current_id + batch_size - 1}")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    self.logger.warning(f"No valid results in batch {current_id}-{current_id + batch_size - 1}. Consecutive failures: {consecutive_failures}")

                current_id += batch_size

            except Exception as e:
                self.logger.error(f"Error processing batch starting at ID {current_id}: {str(e)}", exc_info=True)
                break

        if dataframes:
            self.global_df = pl.concat(dataframes)
            self.logger.info(f"Total games collected: {self.global_df.height}")

        self.control_data.update({
            "first_execution": False,
            "last_id": current_id
        })
        self._save_control_data()

        return self.global_df, self.global_df.height

async def main_bgg(force_restart: bool = False):
    """
    Main entry point for the BGG data collection process. Creates a BGG client instance
    and initiates the continuous scanning process. Handles data persistence with date-stamped
    parquet files.
    Args:
        force_restart (bool, optional): If True, forces the scan to start from the beginning
            and creates a new data file. Defaults to False.
    Returns:
        polars.DataFrame: The complete DataFrame containing all collected game data.
    Example:
        >>> df = await main(force_restart=True)
        >>> print(df.shape)
    Notes:
        - Creates a new file if force_restart is True
        - On first execution, creates a new file regardless of force_restart
        - Otherwise, loads the most recent file and appends new data
    """
    start_time = time.time()

    config = BGGConfig()
    client = BGG(config)
    client.logger.info("Starting BGG data collection process")

    os.makedirs(config.data_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(config.data_dir) if f.startswith(config.base_filename) and f.endswith('.parquet')]

    try:
        if force_restart:
            client.logger.info("Force restart requested - starting fresh data collection")
            df, df_size = await client.continuous_scan(force_restart=True)
            output_file = os.path.join(config.data_dir, f"{config.base_filename}_{client.current_date}.parquet")
            df.write_parquet(output_file)
            client.logger.info(f"Wrote updated data file: {os.path.basename(output_file)}")
        else:
            if not existing_files:
                client.logger.info("No existing data files found - starting fresh collection")
                df, df_size = await client.continuous_scan(force_restart=False)
                output_file = os.path.join(config.data_dir, f"{config.base_filename}_{client.current_date}.parquet")
                df.write_parquet(output_file)
                client.logger.info(f"Wrote updated data file: {os.path.basename(output_file)}")
            else:
                latest_file = max(existing_files, key=lambda x: int(x.split('_')[3].split('.')[0]))
                latest_file_path = os.path.join(config.data_dir, latest_file)
                client.logger.info(f"Loading existing data from {latest_file}")
                existing_df = pl.read_parquet(latest_file_path)
                client.logger.info(f"Existing data contains {existing_df.height} games")

                client.logger.info("Starting collection of new data")
                new_df, df_size = await client.continuous_scan(force_restart=False)
                df = pl.concat([existing_df, new_df])

                output_file = os.path.join(config.data_dir, f"{config.base_filename}_{client.current_date}.parquet")
                df.write_parquet(output_file)
                client.logger.info(f"Wrote updated data file: {os.path.basename(output_file)}")

        execution_time = time.time() - start_time
        client.logger.info(f"Found {df_size} new boardgames")
        client.logger.info(f"Total dataset size: {df.height} boardgames")
        client.logger.info(f"Total execution time: {execution_time / 60:.2f} minutes")

        return df

    except Exception as _:
        client.logger.error("Critical error in main execution", exc_info=True)
        raise
