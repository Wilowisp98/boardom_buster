import requests
import xmltodict
import asyncio
from typing import Dict, Tuple, Union, Optional, List

def make_request(url: str, params: dict) -> requests.Response:
    response = requests.get(url=url, params=params)
    return response

async def api_request(url: str, params: dict) -> requests.Response:
    return await asyncio.to_thread(make_request, url, params)

async def main():
    client = BGGClient()

    tasks = [client.get_game_market_data(i) for i in range(1, 10)]
    result = await asyncio.gather(*tasks)
    return result

class BGGClient:
    def __init__(self, base_url: str = "https://boardgamegeek.com/xmlapi2"):
        self.base_url = base_url

    async def get_game_market_data(self, game_id: int, retry_delay: int = 5, max_retries: int = 3) -> Tuple[str, Dict[str, Union[str, float]], float]:
        """
        Fetches detailed market information for a specific game
        
        Args:
            game_id: BGG game ID
            retry_delay: Seconds to wait between retries
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (game_name, min_price_data, average_price)
        
        Raises:
            Exception: If API request fails or data processing fails
        """
        endpoint = f"{self.base_url}/thing"
        params = {
            "id": game_id,
            "type": "boardgame",
            "marketplace": 1 
        }
        
        response_data = None
        for attempt in range(max_retries):
            response = await api_request(url=endpoint, params=params)
            
            if response.status_code == 200:
                response_data = xmltodict.parse(response.content)
                print(f"Debug: Response data for game ID {game_id}: {response_data}")  # Debug logging
                break
            elif response.status_code == 202:
                print(f"Request queued, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                response.raise_for_status()
        
        if not response_data:
            raise Exception(f"Failed to get response after {max_retries} attempts")
            
        try:
            # Extract game name
            game_name = response_data['items']['item']['name'][0]['@value']
            
            # Process marketplace listings
            min_value = {'link': '', 'value': float('inf')}
            price_list = []
            
            # Handle case where there might be single or multiple listings
            listings = response_data['items']['item'].get('marketplacelistings', {}).get('listing', [])
            if not isinstance(listings, list):
                listings = [listings]
                
            for listing in listings:
                if listing.get('price', {}).get('@currency') != 'EUR':
                    continue
                price = float(listing['price']['@value'])
                price_list.append(price)
                if price < min_value['value']:
                    min_value['link'] = listing.get('link', {}).get('@href', '')
                    min_value['value'] = price

            if not price_list:
                raise ValueError("No EUR prices found in marketplace listings")

            avg_price = sum(price_list) / len(price_list)
            return (game_name, min_value, avg_price)
            
        except KeyError as e:
            raise Exception(f"Failed to process game data: {str(e)}")
        except ValueError as e:
            raise Exception(f"Failed to process pricing data: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error processing game data: {str(e)}")

if __name__ == "__main__":
    try:
        # Check if an event loop is already running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No event loop running
            loop = None

        if loop:
            # If running in an environment with an existing event loop (e.g., Jupyter notebook)
            import nest_asyncio
            nest_asyncio.apply()  # Allow nested event loops
            result = loop.run_until_complete(main())
        else:
            # If running as a standalone script
            result = asyncio.run(main())
        
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")