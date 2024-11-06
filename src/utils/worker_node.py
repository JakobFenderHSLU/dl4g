import logging
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from jass.game.game_observation import GameObservation


class WorkerNode:
    def __init__(self, name, ip, port) -> None:
        self.name = name
        self.ip = ip
        self.port = port

    def ping(self) -> bool:
        """Checks if the worker node is reachable by sending a ping request."""

        url = f"http://{self.ip}:{self.port}/ping"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            logging.debug(f"Error pinging {self.name}: {e}")
            return False

    def process_game_observation(obs: "GameObservation") -> "GameObservation":

        raise NotImplementedError()
