from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation

from play_rule_strategy.abstract_play_rule import PlayRuleStrategy
from play_rule_strategy.mini_max.mini_maxer import MiniMaxer
from play_strategy.nn.mcts.hand_sampler import HandSampler


class MiniMaxPlayRuleStrategy(PlayRuleStrategy):
    def __init__(self, log_level: str, seed: int, depth: int, limit_s: int):
        super().__init__(log_level, __name__, seed)
        self.depth = depth
        self.mini_maxer = MiniMaxer()
        self.hand_sampler = HandSampler()
        self.limit_s = limit_s

    def choose_card(self, obs: GameObservation) -> int | None:
        if sum(obs.hand) > self.depth:
            return None

        # cutoff_time = time.time() + self.limit_s # ignore for now
        children = []
        for i in range(10):
            game_sim = self.__create_game_sim_from_obs(obs)
            root_node = self.mini_maxer.search(game_sim.state)
            children.append(root_node.children)

        cards = [node.card for node in children[0]]
        avg_return = {
            card: sum(node.score for node in children) / len(children)
            for card, children in zip(cards, zip(*children))
        }

        max_card = max(avg_return, key=avg_return.get)

        return max_card

    def __create_game_sim_from_obs(self, game_obs: GameObservation) -> GameSim:
        game_sim = GameSim(rule=self._rule)
        game_sim.init_from_state(state_from_observation(game_obs, HandSampler().sample(game_obs)))
        return game_sim
