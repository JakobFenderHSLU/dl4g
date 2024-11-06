from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import Manager

import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation
from jass.game.rule_schieber import RuleSchieber

from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.nn.mcts.hand_sampler import HandSampler
from src.play_strategy.nn.mcts.mcts_tree import MCTS


class DeterminizedMCTSPlayStrategy(PlayStrategy):
    def __init__(self, limit_s=1, n_threads=16):
        super().__init__(__name__)
        self.n_threads = n_threads
        self.limit_s = limit_s
        self.hand_sampler = HandSampler()
        self.executor = ProcessPoolExecutor()

    def choose_card(self, obs: GameObservation) -> int:
        with Manager() as manager:
            action_scores = manager.Queue()
            valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))

            futures = []
            for _ in range(self.n_threads):
                future = self.executor.submit(_thread_search, action_scores, obs, self.limit_s)
                futures.append(future)

            for future in futures:
                future.result()

            all_action_scores = []
            while not action_scores.empty():
                all_action_scores.append(action_scores.get())

            action_scores = np.array(all_action_scores)
            mean_scores = np.mean(action_scores, axis=0)
            best_card_index = np.argmax(mean_scores)
            return int(valid_cards[best_card_index])


def _thread_search(action_scores, game_obs, limit_s):
    game_sim = __create_game_sim_from_obs(game_obs, RuleSchieber())
    mcts = MCTS()
    mcts.search(game_sim.state, limit_s=limit_s)

    # sort by card index
    mcts.root.children.sort(key=lambda x: x.card)
    action_scores.put([child.score for child in mcts.root.children])


def __create_game_sim_from_obs(game_obs: GameObservation, rule) -> GameSim:
    game_sim = GameSim(rule=rule)
    game_sim.init_from_state(state_from_observation(game_obs, HandSampler().sample(game_obs)))
    return game_sim
