import numpy as np
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

rule = RuleSchieber()


class MiniMaxNode:
    def __init__(
        self, parent: "MiniMaxNode" = None, state: GameState = None, card: int = None
    ):
        self.parent = parent
        self.state = state
        self.card = card
        self.children = []
        self.score = 0
        valid_cards = rule.get_valid_cards_from_state(self.state)
        self.possible_cards = np.where(valid_cards == 1)[0]

    def is_terminal(self):
        return self.state.nr_played_cards == 36

    def evaluate(self):
        return (self.state.points[0] - self.state.points[1]) / 157

    def __repr__(self):
        return f"MiniMaxNode(card={self.card}, score={self.score})"
