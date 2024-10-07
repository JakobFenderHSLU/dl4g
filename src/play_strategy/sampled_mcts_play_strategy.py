import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation

from play_strategy.nn.mcts.mcts_tree import MCTS
from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.nn.mcts.hand_sampler import HandSampler


class SampledMCTSPlayStrategy(PlayStrategy):

    def __init__(self, samples=10, limit_s=1):
        super().__init__(__name__)
        self.samples = samples
        self.limit_s_per_sample = limit_s / samples

    def choose_card(self, obs: GameObservation) -> int:
        action_scores = []
        valid_cards = None
        for i in range(self.samples):
            game_sim = self.__create_game_sim_from_obs(obs)
            mcts = MCTS()
            mcts.search(game_sim.state, limit_s=self.limit_s_per_sample)

            if i == 0:
                valid_cards = mcts.root.possible_cards

            # sort by card index
            mcts.root.children.sort(key=lambda x: x.card)
            action_scores.append([child.score for child in mcts.root.children])

        action_scores = np.array(action_scores)
        mean_scores = np.mean(action_scores, axis=0)
        best_card_index = np.argmax(mean_scores)
        return valid_cards[best_card_index]

    def __create_game_sim_from_obs(self, game_obs: GameObservation) -> GameSim:
        game_sim = GameSim(rule=self._rule)
        game_sim.init_from_state(state_from_observation(game_obs, HandSampler().sample(game_obs)))
        return game_sim
