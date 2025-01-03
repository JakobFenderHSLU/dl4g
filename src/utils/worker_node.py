import asyncio
import logging
import os
import time

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
                async with session.get(url, timeout=int(os.getenv("WORKER_TIMEOUT", 9))) as response:
                    return response.status == 200
        except aiohttp.ClientError as e:
            logging.error(f"Error pinging {self.name}: {e}")
            return False

    async def process_game_observation(self, obs_json: str):
        logging.info(f"Processing game observation on {self.name}")
        start_time = time.time()
        url = f"{self.base_url}/dmcts?obs={obs_json}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=int(os.getenv("WORKER_TIMEOUT", 9))) as response:
                    if response.status == 200:
                        res = await response.json()
                        logging.debug(res)
                        end_time = time.time()
                        total_time = end_time - start_time
                        logging.info(
                            f"Total time for processing game observation on {self.name}: {total_time:.2f} seconds"
                        )
                        return res
                    else:
                        logging.error(
                            f"Failed processing game observation on {self.name}"
                        )
                        return None
        except aiohttp.ClientError as e:
            logging.error(f"Error processing game observation {self.name}: {e}")
            return None
        except asyncio.TimeoutError:
            end_time = time.time()
            total_time = end_time - start_time
            logging.warning(
                f"Task execution for {self.name} exceeded the time limit of 9 seconds ({total_time:.2f})"
            )
            return None
