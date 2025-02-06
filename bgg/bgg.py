import os
import requests
import xmltodict
import asyncio
import json
from typing import List, Optional, Tuple
import polars as pl
from datetime import datetime
import time
from .. import logger

SCHEMA = pl.Schema({
    "game_name": pl.Utf8,
    "description": pl.Utf8,
    "subcategory_1": pl.Utf8,
    "subcategory_2": pl.Utf8,
    "publication_year": pl.Int32,
    "min_players": pl.Int32,
    "max_players": pl.Int32,
    "best_num_players": pl.Int32,
    "recommended_num_players": pl.Int32,
    "suggested_play_age": pl.Int32,
    "categories": pl.List(pl.Utf8),
    "mechanics": pl.List(pl.Utf8),
    "families": pl.List(pl.Utf8),
    "designers": pl.List(pl.Utf8),
    "artists": pl.List(pl.Utf8),
    "publishers": pl.List(pl.Utf8),
    "playing_time": pl.Int32,
    "min_playtime": pl.Int32,
    "max_playtime": pl.Int32,
    "min_age": pl.Int32,
    "language_dependence_description": pl.Utf8,
    "game_rank": pl.Int32,
    "avg_rating": pl.Float64,
    "num_rates": pl.Int32,
    "rank_subcategory_1": pl.Int32,
    "rank_subcategory_2": pl.Int32,
    "avg_weight": pl.Float64,
    "num_weights": pl.Int32,
    "owned_by": pl.Int32,
    "wished_by": pl.Int32
})

class BGG:
    """
    Class for interacting with the BoardGameGeek (BGG) XML API2.

    Attributes:
        base_url (str): The base URL for the BGG API.
    """

    def __init__(self, base_url: str = "https://boardgamegeek.com/xmlapi2") -> None:
        """
        Initializes the BGG class with the base URL.

        Args:
            base_url (str): The base URL for the BGG API. Defaults to "https://boardgamegeek.com/xmlapi2".
        """
        self.base_url = base_url
        self.control_file = "bgg_control.json"
        self.control_data = self._load_control_data()
        self.global_df = None
        self.current_date = int(datetime.now().strftime('%Y%m%d'))
        self.base_filename = "raw_bgg_data"
        self.logger = logger.get_logger('BGGLogger')

    def _load_control_data(self) -> dict:
        if os.path.exists(self.control_file):
            with open(self.control_file, 'r') as f:
                data = json.load(f)
                self.logger.info(f"Loaded control data: last_id={data.get('last_id')}")
                return data
        self.logger.info("No existing control data found, starting fresh")
        return {"first_execution": True, "last_id": 1}

    def _save_control_data(self):
        with open(self.control_file, 'w') as f:
            json.dump(self.control_data, f)

    async def get_game_data(self, game_id: int, retry_delay: int = 5, max_retries: int = 3) -> pl.DataFrame:
        """
        Fetches detailed information for a specific game from the BGG API.

        Args:
            game_id (int): The BGG game ID.
            retry_delay (int): Seconds to wait between retries. Defaults to 5.
            max_retries (int): Maximum number of retry attempts. Defaults to 3.

        Returns:
            pl.DataFrame: A DataFrame containing the processed game data.

        Raises:
            Exception: If the API request fails or data processing fails.
        """
        endpoint = f"{self.base_url}/thing"
        params = {
            "id": game_id,
            "type": "boardgame",
            "stats": 1
        }

        await asyncio.sleep(1)

        response_data = None
        for attempt in range(max_retries):
            try:
                response = await self._make_async_request(url=endpoint, params=params)

                if response.status_code == 200:
                    response_data = xmltodict.parse(response.content)
                    if 'items' not in response_data or not response_data['items'].get('item'):
                        self.logger.warning(f"No data found for game ID {game_id}")
                        response_data = None
                        break
                    self.logger.debug(f"Successfully retrieved data for game ID {game_id}")
                    break
                elif response.status_code == 202:
                    self.logger.info(f"Request queued for game ID {game_id}, attempt {attempt + 1}/{max_retries}")
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"HTTP {response.status_code} for game ID {game_id}")
                    response.raise_for_status()

            except Exception as e:
                self.logger.error(f"Error retrieving game ID {game_id}: {str(e)}", exc_info=True)
                if attempt == max_retries - 1:
                    raise

        if not response_data:
            raise Exception(f"Failed to get response for game ID {game_id} after {max_retries} attempts")

        return self._prepare_data(response_data)

    async def _make_async_request(self, url: str, params: dict) -> requests.Response:
        """
        Makes an asynchronous HTTP GET request.

        Args:
            url (str): The URL to make the request to.
            params (dict): The parameters to include in the request.

        Returns:
            requests.Response: The response from the request.
        """
        return await asyncio.to_thread(self._make_request, url, params)

    def _make_request(self, url: str, params: dict) -> requests.Response:
        """
        Makes a synchronous HTTP GET request.

        Args:
            url (str): The URL to make the request to.
            params (dict): The parameters to include in the request.

        Returns:
            requests.Response: The response from the request.
        """
        return requests.get(url=url, params=params)

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
            for rank in ratings['ranks']['rank']:
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

    def _extract_poll_data(self, game_info: dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[dict]]:
        """
        Extracts poll data from the game information.

        Args:
            game_info (dict): The game information dictionary.

        Returns:
            Tuple[Optional[int], Optional[int], Optional[int], Optional[dict]]: A tuple containing the best number of players,
            recommended number of players, suggested player age, and language dependence information.
        """
        best_numplayers = None
        recommended_numplayers = None
        suggested_playerage = None
        language_dependence = None

        for poll in game_info['poll']:
            if poll['@name'] == 'suggested_numplayers':
                best_votes = 0
                recommended_votes = 0
                for result in poll['results']:
                    for player_result in result.get('result', []):
                        if player_result['@value'] == 'Best' and int(player_result['@numvotes']) > best_votes:
                            best_numplayers = int(result['@numplayers'])
                            best_votes = int(player_result['@numvotes'])

                        if player_result['@value'] == 'Recommended' and int(player_result['@numvotes']) > recommended_votes:
                            recommended_numplayers = int(result['@numplayers'])
                            recommended_votes = int(player_result['@numvotes'])

            elif poll['@name'] == 'suggested_playerage':
                max_votes = 0
                for result in poll['results']['result']:
                    votes = int(result['@numvotes'])
                    if votes > max_votes:
                        suggested_playerage = int(result['@value'])
                        max_votes = votes

            elif poll['@name'] == 'language_dependence':
                max_votes = 0
                for result in poll['results']['result']:
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
        return [link['@value'] for link in game_info['link'] if link['@type'] == link_type]
    
    async def continuous_scan(self, force_restart: bool = False, batch_size: int = 10) -> None:
        """
        Continuously scans and retrieves game data in batches, handling the process in an asynchronous manner.
        Maintains state between executions using control data and combines results into a global DataFrame.

        Args:
            force_restart (bool, optional): If True, resets the control data to start scanning from ID 1.
                Defaults to False.
            batch_size (int, optional): Number of games to process in each batch. Defaults to 10.

        Returns:
            polars.DataFrame: A concatenated DataFrame containing all successfully retrieved game data.

        Notes:
            - Uses control data to track progress between executions
            - Processes games in batches asynchronously
            - Stops after 50 consecutive failed attempts
            - Saves progress periodically through control data
            - Handles exceptions and failed requests gracefully
        """
        if force_restart:
            self.logger.info("Forcing restart of scanning process")
            self.control_data = {"first_execution": True, "last_id": 1}

        start_id = 1 if self.control_data["first_execution"] else self.control_data["last_id"]
        current_id = start_id
        dataframes = []
        consecutive_failures = 0

        self.logger.info(f"Starting continuous scan from ID {start_id}")

        while consecutive_failures < 50:
            try:
                batch_ids = list(range(current_id, current_id + batch_size))
                self.logger.info(f"Processing batch: IDs {current_id} to {current_id + batch_size - 1}")
                
                tasks = [self.get_game_data(game_id) for game_id in batch_ids]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                valid_results = [df for df in results if isinstance(df, pl.DataFrame)]

                if valid_results:
                    dataframes.extend(valid_results)
                    self.logger.info(f"Successfully processed {len(valid_results)} games in batch {current_id}-{current_id + batch_size - 1}")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    self.logger.warning(f"No valid results in batch {current_id}-{current_id + batch_size - 1}. Consecutive failures: {consecutive_failures}")

                current_id += batch_size

            except Exception as e:
                self.logger.error(f"Error processing batch starting at ID {current_id}", exc_info=True)
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
    
    client = BGG()
    client.logger.info("Starting BGG data collection process")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(data_dir) if f.startswith(client.base_filename) and f.endswith('.parquet')]
    
    try:
        if force_restart:
            client.logger.info("Force restart requested - starting fresh data collection")
            df, df_size = await client.continuous_scan(force_restart=True)
            output_file = os.path.join(data_dir, f"{client.base_filename}_{client.current_date}.parquet")
            df.write_parquet(output_file)
            client.logger.info(f"Wrote new data file: {output_file}")
        else:
            if not existing_files:
                client.logger.info("No existing data files found - starting fresh collection")
                df, df_size = await client.continuous_scan(force_restart=False)
                output_file = os.path.join(data_dir, f"{client.base_filename}_{client.current_date}.parquet")
                df.write_parquet(output_file)
                client.logger.info(f"Wrote new data file: {output_file}")
            else:
                latest_file = max(existing_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
                latest_file_path = os.path.join(data_dir, latest_file)
                client.logger.info(f"Loading existing data from {latest_file}")
                existing_df = pl.read_parquet(latest_file_path)
                client.logger.info(f"Existing data contains {existing_df.height} games")
                
                client.logger.info("Starting collection of new data")
                new_df, df_size = await client.continuous_scan(force_restart=False)
                df = pl.concat([existing_df, new_df])
                
                output_file = os.path.join(data_dir, f"{client.base_filename}_{client.current_date}.parquet")
                df.write_parquet(output_file)
                client.logger.info(f"Wrote updated data file: {output_file}")
        
        execution_time = time.time() - start_time
        client.logger.info(f"Found {df_size} new boardgames")
        client.logger.info(f"Total dataset size: {df.height} boardgames")
        client.logger.info(f"Total execution time: {execution_time / 60:.2f} minutes")
        
        return df
        
    except Exception as e:
        client.logger.error("Critical error in main execution", exc_info=True)
        raise