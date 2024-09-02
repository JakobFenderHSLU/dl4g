import time

from jass.game.const import card_ids
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber

from play_strategy.random_play_strategy import RandomPlayStrategy
from src.play_strategy.nn.copied_mcts.hand_sampler import HandSampler
from src.play_strategy.nn.copied_mcts.mcts_node import MCTSNode


class MonteCarloTreeSearch:

    def information_set_search(self, game_state: GameState, iterations=100):
        """
            Determine the best card to play with imperfect information of game state
        :param game_state: The game state with all info of the game
        :param iterations: Iterations of ismcts which should be done
        :return: Best card to play according to ismcts
        """
        sampler = HandSampler()
        rootnode = MCTSNode(state=game_state)

        for i in range(iterations):
            # Selection & Expansion
            node_to_simulate = self._tree_policy(rootnode)

            # sampling
            obs = observation_from_state(node_to_simulate.get_state())
            hands = sampler.sample(game_obs=obs)
            node_to_simulate.randomize_hands(hands)

            # Simulation
            score = self.__simulate_play(node_to_simulate, game_state.player)
            # Backpropagation
            self.__back_propagation(node_to_simulate, score)

        node = self.__get_most_simulated_node(rootnode)
        return card_ids.get(node.get_card())

    def search(self, game_state: GameState, iterations=300, seconds_limit=0.0):
        """
        Determine the card to play after analysis of all hands of the players
        Args:
            game_state: current game state
            iterations: number of iterations to do search
            seconds_limit: If set time which the search will run
        Returns:
            Best card to play according to mcts
        """
        tree = MCTSNode(state=game_state)

        if seconds_limit == 0.0:
            for i in range(iterations):
                # Selection & Expansion
                node_to_simulate = self._tree_policy(tree)
                # Simulation
                score = self.__simulate_play(node_to_simulate, game_state.player)
                # Backpropagation
                self.__back_propagation(node_to_simulate, score)
        else:
            cut_off_time = time.time() + seconds_limit
            while time.time() < cut_off_time:
                node_to_simulate = self._tree_policy(tree)
                # Simulation
                score = self.__simulate_play(node_to_simulate, game_state.player)
                # Backpropagation
                self.__back_propagation(node_to_simulate, score)

        node = self.__get_most_simulated_node(tree)
        return card_ids.get(node.get_card()), node.get_simulation_cnt()

    def _tree_policy(self, node: MCTSNode):
        """
        :param node: root node from which to find best next best one
        :return: most promising node to explore
        -------
        """
        current_node = node
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child_ubc()
        return current_node

    def __simulate_play(self, node: MCTSNode, simulating_player: int):
        """
        :param node: selected node which simulated
        :param simulating_player: player which started mcts simulation
        :return: outcome of play
        -------
        """
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_state(node.get_state())

        random_play_strategy = RandomPlayStrategy()

        while not game_sim.is_done():
            game_sim.action_play_card(random_play_strategy.choose_card(game_sim.get_observation()))

        if simulating_player % 2 == 0:
            # normalize points
            return game_sim.state.points[0] / sum(game_sim.state.points)
        else:
            # normalize points
            return game_sim.state.points[1] / sum(game_sim.state.points)

    def __back_propagation(self, node: MCTSNode, score: int):
        """
        :param node: node which payoff was calculated
        :param score: the score that was achieved in simulation
        -------
        """
        tmp_node = node
        while tmp_node is not None:
            tmp_node.increment_simulation_cnt()
            tmp_node.set_win_score(score)
            tmp_node = tmp_node.get_parent()

    def __get_most_simulated_node(self, tree: MCTSNode):
        most_simulated = MCTSNode()
        child: MCTSNode
        for child in tree.get_children():
            if child.get_simulation_cnt() > most_simulated.get_simulation_cnt():
                most_simulated = child
        return most_simulated
