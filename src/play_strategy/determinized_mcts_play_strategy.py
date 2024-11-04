import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation

from play_strategy.nn.mcts.mcts_tree import MCTS
from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.nn.mcts.hand_sampler import HandSampler


class DeterminizedMCTSPlayStrategy(PlayStrategy):

    def __init__(self, samples=10, limit_s=1, n_threads=16):
        super().__init__(__name__)
        self.samples = samples
        self.n_threads = n_threads
        self.limit_s = limit_s
        self.hand_sampler = HandSampler()
        self.executor = ThreadPoolExecutor(max_workers=self.n_threads)

    def choose_card(self, obs: GameObservation) -> int:
        action_scores = queue.Queue()
        valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))

        futures = [
            self.executor.submit(self.__thread_search, action_scores, obs)
            for _ in range(self.n_threads)
        ]

        for future in futures:
            future.result()

        action_scores = list(action_scores.queue)
        action_scores = np.array(action_scores)
        mean_scores = np.mean(action_scores, axis=0)
        best_card_index = np.argmax(mean_scores)
        return int(valid_cards[best_card_index])

    def __thread_search(self, action_scores, game_obs):
        game_sim = self.__create_game_sim_from_obs(game_obs)
        mcts = MCTS()
        mcts.search(game_sim.state, limit_s=self.limit_s)

        # sort by card index
        mcts.root.children.sort(key=lambda x: x.card)
        action_scores.put([child.score for child in mcts.root.children])

    def __create_game_sim_from_obs(self, game_obs: GameObservation) -> GameSim:
        game_sim = GameSim(rule=self._rule)
        game_sim.init_from_state(state_from_observation(game_obs, self.hand_sampler.sample(game_obs)))
        return game_sim
