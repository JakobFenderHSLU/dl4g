import json
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.nn.mcts.dmcts_worker import DMCTSWorker
from src.play_strategy.nn.mcts.hand_sampler import HandSampler
from src.utils.worker_node_manager import WorkerNodeManager


class DeterminizedMCTSPlayStrategy(PlayStrategy):
    def __init__(self, limit_s):
        super().__init__(__name__)
        self.limit_s = limit_s
        self.hand_sampler = HandSampler()
        self.executor = ProcessPoolExecutor()
        self.dmcts_worker = DMCTSWorker(limit_s)
        self.worker_node_manager = WorkerNodeManager()

    def choose_card(self, obs: GameObservation) -> int:
        valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))

        obs_json_str = json.dumps(obs.to_json())
        action_scores = self.worker_node_manager.execute_all_dmcts(obs_json_str)

        mean_scores = np.mean(action_scores, axis=0)
        best_card_index = np.argmax(mean_scores)
        return int(valid_cards[best_card_index])
