import queue
import time
from concurrent.futures.thread import ThreadPoolExecutor

from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation

from play_rule_strategy.abstract_play_rule import PlayRuleStrategy
from play_rule_strategy.mini_max.mini_maxer import MiniMaxer
from play_strategy.nn.mcts.hand_sampler import HandSampler
from play_strategy.random_play_strategy import RandomPlayStrategy


class MiniMaxPlayRuleStrategy(PlayRuleStrategy):
    def __init__(
        self, log_level: str, seed: int, depth: int, limit_s: float, n_threads: int = 32
    ):
        super().__init__(log_level, __name__, seed)
        self.depth = depth
        self.mini_maxer = MiniMaxer()
        self.hand_sampler = HandSampler()
        self.limit_s = limit_s
        self.n_threads = n_threads
        self.executor = ThreadPoolExecutor(max_workers=self.n_threads)
        self.fallback = RandomPlayStrategy()

    def choose_card(self, obs: GameObservation) -> int | None:
        if sum(obs.hand) > self.depth:
            return None

        cutoff_time = time.time() + self.limit_s
        children = queue.Queue()

        futures = [
            self.executor.submit(self.__thread_search, children, obs, cutoff_time)
            for _ in range(self.n_threads)
        ]

        for future in futures:
            future.result()

        children = list(children.queue)

        if len(children) == 0:
            # Note (Jakob): if no children were generated in time. Choose random card, since we have no time
            # budget left to compute something more advanced.
            return self.fallback.choose_card(obs)

        cards = [node.card for node in children[0]]
        avg_return = {
            card: sum(node.score for node in children) / len(children)
            for card, children in zip(cards, zip(*children))
        }

        max_card = max(avg_return, key=avg_return.get)

        return max_card

    def __thread_search(self, children, game_obs, cutoff_time):
        while time.time() < cutoff_time:
            game_sim = self.__create_game_sim_from_obs(game_obs)
            root_node = self.mini_maxer.search(game_sim.state, cutoff_time=cutoff_time)
            if root_node is not None:
                children.put(root_node.children)

    def __create_game_sim_from_obs(self, game_obs: GameObservation) -> GameSim:
        game_sim = GameSim(rule=self._rule)
        game_sim.init_from_state(
            state_from_observation(game_obs, HandSampler().sample(game_obs))
        )
        return game_sim
