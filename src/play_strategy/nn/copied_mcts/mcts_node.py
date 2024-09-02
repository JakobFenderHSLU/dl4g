from __future__ import annotations  # import to allow self referencing type

import copy

import numpy as np
from jass.game.const import card_ids
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_str_encoded_list
from jass.game.rule_schieber import RuleSchieber


class MCTSNode:
    _children: []
    _state: GameState
    _parent: MCTSNode
    _card: str
    _win_score: float
    _simulation_cnt: int

    def __init__(self, parent=None, state=None, card=None):
        self._children = []
        self._parent = parent
        self._state = state
        self._card = card
        self._win_score = 0.0
        self._simulation_cnt = 0
        player_hand = []
        if self._state is not None:
            player_hand = RuleSchieber().get_valid_cards(self._state.hands[self._state.player],
                                                         self._state.current_trick,
                                                         self._state.nr_cards_in_trick,
                                                         self._state.trump)

        self._available_cards = convert_one_hot_encoded_cards_to_str_encoded_list(player_hand)
        self._available_cards_cnt = len(self._available_cards)

    def is_terminal_node(self):
        return self._state.nr_played_cards == 36

    def is_fully_expanded(self):
        return len(self._children) == self._available_cards_cnt

    def best_child_ubc(self, c=np.sqrt(2)):
        choices_weights = [
            (child.get_win_score() / child.get_simulation_cnt()) + c * np.sqrt(
                (2 * np.log(self.get_simulation_cnt()) / child.get_simulation_cnt()))
            for child in self._children
        ]
        return self._children[np.argmax(choices_weights)]

    def expand(self):
        node_sim = GameSim(rule=RuleSchieber())
        node_sim.init_from_state(self.get_state())

        card = np.random.choice(self.get_available_cards())
        self.remove_available_card(card)

        sim_copy = copy.deepcopy(node_sim)
        sim_copy.action_play_card(card_ids.get(card))
        new_node = MCTSNode(self, sim_copy.state, card)
        new_node.set_parent(self)
        self.add_child(new_node)
        return new_node

    def randomize_hands(self, hands: np.array):
        self._state.hands = hands

    # GETTERS & SETTERS
    def get_parent(self):
        return self._parent

    def set_parent(self, parent):
        self._parent = parent

    def get_children(self) -> []:
        return self._children

    def add_child(self, child):
        self._children.append(child)

    def get_simulation_cnt(self) -> int:
        return self._simulation_cnt

    def increment_simulation_cnt(self):
        self._simulation_cnt += 1

    def get_state(self) -> GameState:
        return self._state

    def get_win_score(self) -> float:
        return self._win_score

    def set_win_score(self, score: float):
        self._win_score += score

    def get_card(self):
        return self._card

    def remove_available_card(self, card):
        self._available_cards.remove(card)

    def get_available_cards(self):
        return self._available_cards

    def __str__(self):
        return "Node for card: " + str(self._card) + ", Cards played: " + str(self._state.nr_played_cards)
