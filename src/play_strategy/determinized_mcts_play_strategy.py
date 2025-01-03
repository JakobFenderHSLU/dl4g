import json
import logging
import time

import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.nn.mcts.hand_sampler import HandSampler
from src.utils.worker_node_manager import WorkerNodeManager


class DeterminizedMCTSPlayStrategy(PlayStrategy):
    def __init__(self, limit_s):
        super().__init__(__name__)
        self.limit_s = limit_s
        self.hand_sampler = HandSampler()

    def choose_card(self, obs: GameObservation) -> int:
        start_time = time.time()
        valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))

        obs_json_str = json.dumps(obs.to_json())
        worker_node_manager = WorkerNodeManager()
        # shape: (n_nodes, n_determinisations_per_node, valid_cards_score)
        action_scores = worker_node_manager.execute_all_dmcts(obs_json_str)
        if len(action_scores) == 0:
            logging.error("No action scores returned")
            return int(valid_cards[0])
        # shape: (n_determinisations, valid_cards_score)
        action_scores = np.concatenate(action_scores)
        logging.debug("action_scores")
        logging.debug(action_scores)

        try:
            mean_scores = np.mean(action_scores, axis=0)
            logging.debug("mean_scores")
            logging.debug(mean_scores)
            best_card_index = np.argmax(mean_scores)
            logging.debug("best_card_index")
            logging.debug(best_card_index)
            best_card = int(valid_cards[best_card_index])
            logging.debug("best_card")
            logging.debug(best_card)
        except Exception as e:
            logging.error("Error in choose_card: %s", str(e))
            best_card = int(valid_cards[0])

        execution_time = time.time() - start_time

        if execution_time > 9.5:
            logging.error(
                f"Execution time choose_card() exceeded 9.5 seconds: {execution_time:.2f} seconds"
            )
        else:
            logging.info(f"Execution time choose_card(): {execution_time:.2f} seconds")
        return best_card
