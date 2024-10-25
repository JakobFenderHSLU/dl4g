import copy

import numpy as np
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

rule = RuleSchieber()

np_sqrt_2 = np.sqrt(2)


class MCTSNode:
    def __init__(
        self, parent: "MCTSNode" = None, state: GameState = None, card: int = None
    ):
        self.parent = parent
        self.state = state
        self.card = card
        self.score = 0.0
        self.n_simulated = 0
        valid_cards = rule.get_valid_cards(
            hand=self.state.hands[self.state.player],
            current_trick=self.state.current_trick,
            move_nr=self.state.nr_cards_in_trick,
            trump=self.state.trump,
        )
        self.possible_cards = np.where(valid_cards == 1)[0]
        self.not_simulated_cards = self.possible_cards.copy()
        self.n_possible_cards = len(self.possible_cards)
        self.children = []
        self.children_scores = np.ones(self.n_possible_cards) * -1

        self.is_terminal = self.state.nr_played_cards == 36

    @property
    def is_fully_expanded(self):
        return len(self.children) == self.n_possible_cards

    def best_child_ubc(self, c=np_sqrt_2):  # TODO: WRITE OWN IMPLEMENTATION
        choices_weights = [
            (child.score / child.n_simulated)
            + c * np.sqrt((2 * np.log(self.n_simulated) / child.n_simulated))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        node_sim = GameSim(rule=rule)
        node_sim.init_from_state(self.state)

        card = np.random.choice(self.not_simulated_cards)
        index_of_card = np.where(self.possible_cards == card)[0][0]
        self.not_simulated_cards = np.delete(
            self.not_simulated_cards, np.where(self.not_simulated_cards == card)
        )

        sim_copy = copy.deepcopy(node_sim)
        sim_copy.action_play_card(card)
        new_node = MCTSNode(self, sim_copy.state, card)
        new_node.parent = self
        self.children.append(new_node)
        self.children_scores[index_of_card] = 0
        return new_node

    def __repr__(self):
        return f"MCTSNode(card={self.card}, score={self.score}, n_simulated={self.n_simulated})"
