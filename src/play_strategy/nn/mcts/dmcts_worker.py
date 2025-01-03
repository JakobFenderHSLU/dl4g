import logging
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation
from jass.game.rule_schieber import RuleSchieber

from src.play_strategy.nn.mcts.hand_sampler import HandSampler
from src.play_strategy.nn.mcts.mcts_tree import MCTS


class DMCTSWorker:
    def __init__(self, limit_s: float, n_determinations: int = None, n_iterations: int = None):
        self._rule = RuleSchieber()
        self.limit_s = limit_s
        self.n_determinations = n_determinations
        self.n_iterations = n_iterations
        self.executor = ProcessPoolExecutor()
        self.manager = Manager()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DMCTSWorker initialized with limit_s={limit_s}, n_determinations={n_determinations}, "
                         f"n_iterations={n_iterations}")

    def execute(self, obs: GameObservation, n_determinations: int = None) -> np.ndarray:
        """
        Execute the determinized MCTS search. This method will return the action scores for each card in the hand.
        :param obs: The game observation
        :param n_determinations: The number of determinations to run in parallel. If None, the number of logical CPUs
        :return: The action scores for each card in the hand
        """
        action_scores = self.manager.Queue()

        start_time = time.time()
        futures = []
        for _ in range(self.n_determinations):
            future = self.executor.submit(
                _thread_search, action_scores, obs, self.limit_s, self.n_iterations
            )
            futures.append(future)

        for future in futures:
            future.result()

        execution_time = time.time() - start_time
        self.logger.info(
            f"It took {execution_time:.3f} seconds to run the determinations"
        )

        all_action_scores = []
        while not action_scores.empty():
            all_action_scores.append(action_scores.get())

        action_scores = np.array(all_action_scores)
        return action_scores


def _thread_search(action_scores, game_obs, limit_s, n_iterations):
    """
    Thread function to run the MCTS search
    :param action_scores: A queue to put the action scores in
    :param game_obs: The game observation
    :param limit_s: The time limit for the search
    :return: None
    """

    game_sim = GameSim(rule=RuleSchieber())
    game_sim.init_from_state(
        state_from_observation(game_obs, HandSampler().sample(game_obs))
    )
    mcts = MCTS()
    mcts.search(game_sim.state, limit_s=limit_s, iterations=n_iterations)

    # sort by card index
    mcts.root.children.sort(key=lambda x: x.card)
    action_scores.put([child.score for child in mcts.root.children])
