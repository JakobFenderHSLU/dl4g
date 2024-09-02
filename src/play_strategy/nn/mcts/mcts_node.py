import copy

import numpy as np
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

rule = RuleSchieber()


class MCTSNode:
    def __init__(self, parent: "MCTSNode" = None, state: GameState = None, card: int = None):
        self.children = []
        self.parent = parent
        self.state = state
        self.card = card
        self.score = 0.0
        self.n_simulated = 0
        valid_cards = rule.get_valid_cards(
            hand=self.state.hands[self.state.player],
            current_trick=self.state.current_trick,
            move_nr=self.state.nr_cards_in_trick,
            trump=self.state.trump
        )
        self.possible_cards = np.where(valid_cards == 1)[0]
        self.n_possible_cards = len(self.possible_cards)

        self.is_terminal = self.state.nr_played_cards == 36

    @property
    def is_fully_expanded(self):
        return len(self.children) == self.n_possible_cards

    def best_child_ubc(self, c=np.sqrt(2)):  # TODO: WRITE OWN IMPLEMENTATION
        choices_weights = [
            (child.score / child.n_simulated) + c * np.sqrt(
                (2 * np.log(self.n_simulated) / child.n_simulated))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        node_sim = GameSim(rule=RuleSchieber())
        node_sim.init_from_state(self.state)

        card = np.random.choice(self.possible_cards)
        self.possible_cards = np.delete(self.possible_cards, np.where(self.possible_cards == card))

        sim_copy = copy.deepcopy(node_sim)
        sim_copy.action_play_card(card)
        new_node = MCTSNode(self, sim_copy.state, card)
        new_node.parent = self
        self.children.append(new_node)
        return new_node
