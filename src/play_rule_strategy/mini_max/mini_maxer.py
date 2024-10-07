from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber

from play_rule_strategy.mini_max.mini_max_node import MiniMaxNode

rule = RuleSchieber()


class MiniMaxer:
    def __init__(self):
        pass

    def search(self, game_state, limit_s=None) -> MiniMaxNode:
        root = MiniMaxNode(parent=None, state=game_state)
        self.minimax(root, float('-inf'), float('inf'), True)
        return root

    def minimax(self, node, alpha, beta, maximizing_team):
        if node.is_terminal():
            return node.evaluate()

        if maximizing_team:
            max_eval = float('-inf')
            for possible_card in node.possible_cards:
                game_sim = GameSim(rule=rule)
                game_sim.init_from_state(node.state)
                game_sim.action_play_card(possible_card)
                child_state = game_sim.state

                child = MiniMaxNode(parent=node, state=child_state, card=possible_card)
                node.children.append(child)
                eval = self.minimax(child, alpha, beta, False)
                max_eval = max(max_eval, eval)
                node.score = max_eval
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = float('inf')
            for possible_card in node.possible_cards:
                game_sim = GameSim(rule=rule)
                game_sim.init_from_state(node.state)
                game_sim.action_play_card(possible_card)
                child_state = game_sim.state

                child = MiniMaxNode(parent=node, state=child_state, card=possible_card)
                node.children.append(child)
                eval = self.minimax(child, alpha, beta, True)
                min_eval = min(min_eval, eval)
                node.score = min_eval
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
