import time

from jass.game.const import card_strings
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber

from src.play_rule_strategy.mini_max.mini_max_node import MiniMaxNode

rule = RuleSchieber()


class MiniMaxer:
    def __init__(self):
        self.cutoff_time = None

    def search(self, game_state, cutoff_time=None) -> MiniMaxNode:
        self.cutoff_time = cutoff_time
        if not cutoff_time:
            self.cutoff_time = float('inf')

        root = MiniMaxNode(parent=None, state=game_state)
        self.minimax(root, float('-inf'), float('inf'), True)
        if self.cutoff_time <= time.time():
            return None
        return root

    def minimax(self, node, alpha, beta, maximizing_team):
        if self.cutoff_time <= time.time():
            return None

        if node.is_terminal():
            return node.evaluate()

        if maximizing_team:
            max_eval = float("-inf")
            for possible_card in node.possible_cards:
                game_sim = GameSim(rule=rule)
                game_sim.init_from_state(node.state)
                game_sim.action_play_card(possible_card)
                child_state = game_sim.state

                child = MiniMaxNode(parent=node, state=child_state, card=possible_card)
                node.children.append(child)
                eval = self.minimax(child, alpha, beta, False)
                if eval is None:
                    return None
                max_eval = max(max_eval, eval)
                node.score = max_eval
                alpha = max(alpha, eval)
            return max_eval
        else:
            min_eval = float("inf")
            for possible_card in node.possible_cards:
                game_sim = GameSim(rule=rule)
                game_sim.init_from_state(node.state)
                game_sim.action_play_card(possible_card)
                child_state = game_sim.state

                child = MiniMaxNode(parent=node, state=child_state, card=possible_card)
                node.children.append(child)
                eval = self.minimax(child, alpha, beta, True)
                if eval is None:
                    return None
                min_eval = min(min_eval, eval)
                node.score = min_eval
                beta = min(beta, eval)
            return min_eval

    def print_tree(self, root, depth=100):
        def print_tree_node(node, node_depth):
            display_name = card_strings[node.card] if node.card is not None else "ROOT"
            print("    " * node_depth, f"{display_name}: {node.score}")
            if node_depth < depth:
                for child in node.children:
                    print_tree_node(child, node_depth + 1)

        print_tree_node(root, 0)
