import logging

import requests


class WorkerNode:
    def __init__(self, name, ip, port) -> None:
        self.name = name
        self.ip = ip
        self.port = port
        self.base_url = f"http://{self.ip}:{self.port}"

    def ping(self) -> bool:
        """Checks if the worker node is reachable by sending a ping request."""

        url = f"{self.base_url}/ping"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            logging.debug(f"Error pinging {self.name}: {e}")
            return False

    async def process_game_observation(self, obs_json: str):
        url = f"{self.base_url}/dmcts?obs={obs_json}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logging.debug(response.json())
                return response.json()
            else:
                return None
        except requests.exceptions.RequestException as e:
            logging.debug(f"Error processing game observation {self.name}: {e}")
            return None
