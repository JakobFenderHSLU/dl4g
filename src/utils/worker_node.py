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
                async with session.get(url, timeout=10) as response:
                    return response.status == 200
        except aiohttp.ClientError as e:
            logging.error(f"Error pinging {self.name}: {e}")
            return False

    async def process_game_observation(self, obs_json: str):
        logging.debug("processing game observation")
        url = f"{self.base_url}/dmcts?obs={obs_json}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=9) as response:
                    if response.status == 200:
                        res = await response.json()
                        logging.debug(res)
                        return res
                    else:
                        return None
        except aiohttp.ClientError as e:
            logging.error(f"Error processing game observation {self.name}: {e}")
            return None
        except asyncio.TimeoutError:
            logging.warning(
                f"Task execution for {self.name} exceeded the time limit of 9 seconds"
            )
            return None
