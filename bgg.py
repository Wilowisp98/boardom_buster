import requests
import xmltodict
import asyncio
from typing import List, Optional, Tuple
import polars as pl

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
            "type": "boardgame"
        }

        response_data = None
        for _ in range(max_retries):
            response = await self._make_async_request(url=endpoint, params=params)

            if response.status_code == 200:
                response_data = xmltodict.parse(response.content)
                break
            elif response.status_code == 202:
                print(f"Request queued, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                response.raise_for_status()

        if not response_data:
            raise Exception(f"Failed to get response after {max_retries} attempts")

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

            # Create DataFrame
            df = pl.DataFrame({
                "game_name": [game_name],
                "description": [game_description],
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
                "language_dependence_level": [language_dependence['level'] if language_dependence else None],
                "language_dependence_description": [language_dependence['value'] if language_dependence else None]
            })

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
                        language_dependence = {
                            'level': int(result['@level']),
                            'value': result['@value']
                        }
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

async def main(game_ids: List[int]) -> pl.DataFrame:
    """
    Fetches game data for a list of game IDs asynchronously and returns a combined DataFrame.

    Args:
        game_ids (List[int]): A list of BGG game IDs.

    Returns:
        pl.DataFrame: A DataFrame containing the combined data for all requested games.
    """
    client = BGGClient()
    tasks = [client.get_game_data(game_id) for game_id in game_ids]
    results = await asyncio.gather(*tasks)
    return pl.concat(results)

if __name__ == "__main__":
    game_ids = [1, 2, 3, 4, 5]
    df = asyncio.run(main(game_ids))
    print(df)