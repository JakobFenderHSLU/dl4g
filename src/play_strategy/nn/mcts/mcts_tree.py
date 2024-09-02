import time

from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

from play_strategy.nn.mcts.mcts_node import MCTSNode
from play_strategy.random_play_strategy import RandomPlayStrategy


class MCTS:

    def __init__(self):
        self.root = None

    def search(self, game_state: GameState, iterations=100, limit_s=None):

        self.root = MCTSNode(state=game_state)

        if limit_s is None:
            for i in range(iterations):
                self._run(game_state)
        else:
            cutoff_time = time.time() + limit_s
            while time.time() < cutoff_time:
                self._run(game_state)

        return self._get_most_simulated_node().card

    def _run(self, game_state: GameState):
        # get the best node to simulate
        node = self._tree_policy(self.root)

        # simulate the game
        score = self._simulate(node, game_state.player)

        # back propagate the score
        self._back_propagation(node, score)

    def _tree_policy(self, node: MCTSNode):
        current_node = node
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child_ubc()
        return current_node

    def _simulate(self, node: MCTSNode, simulating_player: int):
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_state(node.state)

        random_play_strategy = RandomPlayStrategy()

        while not game_sim.is_done():
            # get obs from state
            obs = game_sim.get_observation()
            card = random_play_strategy.choose_card(obs)
            game_sim.action_play_card(card)

        if simulating_player % 2 == 0:
            return game_sim.state.points[0] / 157
        else:
            return game_sim.state.points[1] / 157

    @staticmethod
    def _back_propagation(node: MCTSNode, score: int):
        while node is not None:
            node.n_simulated += 1
            node.score += score
            node = node.parent

    def _get_most_simulated_node(self):
        return max(self.root.children, key=lambda child: child.n_simulated)
