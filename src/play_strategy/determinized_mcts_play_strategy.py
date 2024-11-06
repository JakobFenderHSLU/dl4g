import json
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
import requests
from jass.game.game_observation import GameObservation

from play_strategy.nn.mcts.dmcts_worker import DMCTSWorker
from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.nn.mcts.hand_sampler import HandSampler


class DeterminizedMCTSPlayStrategy(PlayStrategy):
    def __init__(self, limit_s):
        super().__init__(__name__)
        self.limit_s = limit_s
        self.hand_sampler = HandSampler()
        self.executor = ProcessPoolExecutor()
        self.dmcts_worker = DMCTSWorker(limit_s)

    def choose_card(self, obs: GameObservation) -> int:
        valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))

        obs_json_str = json.dumps(obs.to_json())

        # EXAMPLE CODE
        # TODO: @Dan: Implement the following code with the nodes asynchronously
        action_scores = []
        node_list = []
        for node in node_list:
            response = requests.get(f"{node.url}/dmcts?obs={obs_json_str}")
            action_scores.append(response.json())

        # END EXAMPLE CODE

        # Current code to keep the functionality for now
        action_scores = self.dmcts_worker.execute(obs)

        mean_scores = np.mean(action_scores, axis=0)
        best_card_index = np.argmax(mean_scores)
        return int(valid_cards[best_card_index])
