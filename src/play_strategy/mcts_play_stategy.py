from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation

from play_strategy.nn.mcts.mcts_tree import MCTS
from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.nn.mcts.hand_sampler import HandSampler


class MCTSPlayStrategy(PlayStrategy):

    def __init__(self):
        super().__init__(__name__)

    def choose_card(self, obs: GameObservation) -> int:
        game_sim = self.__create_game_sim_from_obs(obs)
        return MCTS().search(game_sim.state, iterations=1000)

    def __create_game_sim_from_obs(self, game_obs: GameObservation) -> GameSim:
        game_sim = GameSim(rule=self._rule)
        game_sim.init_from_state(state_from_observation(game_obs, HandSampler().sample(game_obs)))
        return game_sim
