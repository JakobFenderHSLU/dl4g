import math
import time

import numpy as np
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

from src.play_strategy.nn.mcts.mcts_node import MCTSNode
from src.play_strategy.random_play_strategy import RandomPlayStrategy

np_sqrt_2 = np.sqrt(2)


class MCTS:
    def __init__(self, ucb_c=np_sqrt_2):
        self.root = None
        self.ucb_c = ucb_c

    def search(self, game_state: GameState, iterations=None, limit_s=None, root=None):

        if root is not None:
            self.root = root
        else:
            self.root = MCTSNode(state=game_state, ucb_c=self.ucb_c)

        if iterations is None and limit_s is None:
            raise ValueError("Either iterations or limit_s must be set")

        if iterations is not None:
            for i in range(iterations):
                is_fully_expanded = self._run(game_state)
                if is_fully_expanded:
                    break
        if limit_s is not None:
            cutoff_time = time.time() + limit_s
            while time.time() < cutoff_time:
                is_fully_expanded = self._run(game_state)
                if is_fully_expanded:
                    break

        return self._get_most_simulated_node().card

    def _run(self, game_state: GameState) -> bool:
        # get the best node to simulate
        node = self._tree_policy(self.root)
        if node is None:
            return True

        # simulate the game
        score = self._simulate(node, game_state.player)

        # back propagate the score
        self._back_propagation(node, score)
        return False

    def _tree_policy(self, node: MCTSNode) -> MCTSNode or None:
        """
        Traverse the tree to find the best node to simulate
        :param node: The node to start the tree policy from
        :return: The best node to simulate
        """
        current_node = node
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child_ubc()
        if current_node.score != -math.inf:
            return None
        return current_node

    def _simulate(self, node: MCTSNode, simulating_player: int) -> float:
        """
        Simulate a game from the current node
        :param node: The node to simulate from containing a valid game state
        :param simulating_player: The player to simulate for
        :return: The score of the simulated game
        """
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_state(node.state)

        random_play_strategy = RandomPlayStrategy()

        while not game_sim.is_done():
            obs = game_sim.get_observation()
            card = random_play_strategy.choose_card(obs)
            game_sim.action_play_card(card)

        return (
            game_sim.state.points[simulating_player % 2]
            - game_sim.state.points[(simulating_player + 1) % 2]
        ) / 157

    @staticmethod
    def _back_propagation(node: MCTSNode, score: float) -> None:
        """
        Back propagate the score of the simulated game
        :param node: The node to back propagate from
        :param score: The score of the simulated game
        """
        prev_node = None
        while node is not None:
            node.n_simulated += 1
            if prev_node is None:
                node.score = score
            else:
                prev_node_card = prev_node.card
                prev_node_index = node.possible_cards.tolist().index(prev_node_card)
                node.children_scores[prev_node_index] = score
                children_scores = node.children_scores[node.children_scores != -1]
                node.score = sum(children_scores) / len(node.children_scores)

            prev_node = node
            node = node.parent

    def _get_most_simulated_node(self):
        return max(self.root.children, key=lambda child: child.n_simulated)
