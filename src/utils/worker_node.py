import asyncio
import logging

import aiohttp


class WorkerNode:
    def __init__(self, name, ip, port) -> None:
        self.name = name
        self.ip = ip
        self.port = port
        self.base_url = f"http://{self.ip}:{self.port}"

    async def ping(self) -> bool:
        """Checks if the worker node is reachable by sending a ping request."""
        url = f"{self.base_url}/ping"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return response.status == 200
        except aiohttp.ClientError as e:
            logging.debug(f"Error pinging {self.name}: {e}")
            return False

    async def process_game_observation(self, obs_json: str):
        logging.debug("processing game observation")
        url = f"{self.base_url}/dmcts?obs={obs_json}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=9.5) as response:
                    if response.status == 200:
                        logging.debug(await response.json())
                        return await response.json()
                    else:
                        return None
        except aiohttp.ClientError as e:
            logging.debug(f"Error processing game observation {self.name}: {e}")
            return None
        except asyncio.TimeoutError:
            logging.warning(
                f"Task execution for {self.name} exceeded the time limit of 9.5 seconds"
            )
            return None
