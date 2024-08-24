import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from jass.game.const import MAX_TRUMP
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber
from numpy import ndarray

import src.utils.game_utils as gu


class TrumpDataGenerator:
    def __init__(self, load_data=False, n_play_per_hand=100, backup_interval=1000, verbose=False):
        self.existing_deck = set()
        self.n_yielded_hands = 0
        self.deck = np.arange(36)
        self.n_play_per_hand = n_play_per_hand
        self.backup_interval = backup_interval
        self.verbose = verbose

        self.game = GameSim(RuleSchieber())
        self.game_state = GameState()
        self.game_state.dealer = 0
        self.game_state.player = 1

        self.cached_decks = pd.DataFrame(columns=[
            "hand", "deck", "score"
        ])

        if load_data:
            if os.path.exists("data/existing_hands.pickle"):
                with open("data/existing_hands.pickle", "rb") as f:
                    self.existing_deck = pickle.load(f)

            # if os.path.exists("data/cached_hands.csv"):
            #     with open("data/cached_hands.csv", "r") as f:
            #         self.cached_decks = pd.read_csv(f)

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_yielded_hands % self.backup_interval == 0 and self.n_yielded_hands > 0:
            print(f"Backing up hands at {self.n_yielded_hands}...")
            self._backup_hands()

        if self.n_yielded_hands < len(self.cached_decks):
            print(f"Yielding cached hand {self.n_yielded_hands}...")
            hand = self.cached_decks.iloc[self.n_yielded_hands][:-1].values
            self.n_yielded_hands += 1
            return hand

        deck = self._generate_unique_deck()
        verbose_results = self._get_scores(deck)
        result = np.concatenate([deck.flatten(), verbose_results.flatten()])

        # cache the result
        # self.cached_decks.loc[self.n_yielded_hands] = result
        self.n_yielded_hands += 1

        return result

    def _backup_hands(self):
        with open("data/existing_hands.pickle", "wb") as f:
            pickle.dump(self.existing_deck, f)

        # with open("data/cached_hands.csv", "a") as f:
        #     self.cached_decks.to_csv(f, index=False, header=False)

    def _generate_unique_deck(self) -> np.ndarray:
        deck, deck_str = self._generate_random_deck()

        while deck_str in self.existing_deck:
            deck, deck_str = self._generate_random_deck()

        self.existing_deck.add(deck_str)
        return deck

    def _generate_random_deck(self) -> Tuple[np.ndarray, str]:
        deck = self.deck.copy()
        np.random.shuffle(deck)
        hands = gu.deck_to_onehot_hands(deck)
        swap_hand, color_order = gu.swap_colors(hands[0])
        swapped_deck = np.array([
            swap_hand,
            gu.swap_colors_from_order(hands[1], color_order),
            gu.swap_colors_from_order(hands[2], color_order),
            gu.swap_colors_from_order(hands[3], color_order)
        ])

        deck_str = "".join(swapped_deck.flatten().astype(str))
        return deck, deck_str

    def _get_scores(self, deck: np.ndarray) -> ndarray:
        self.game_state.hands = gu.deck_to_onehot_hands(deck)

        self.game.init_from_state(self.game_state)

        results = np.zeros((MAX_TRUMP + 1, self.n_play_per_hand))

        for trump in range(MAX_TRUMP + 1):
            score = np.zeros(self.n_play_per_hand)
            for i in range(self.n_play_per_hand):
                self.game_state.trump = trump
                self.game.init_from_state(self.game_state)

                self.game.action_trump(trump)

                for cards in range(36):
                    obs = observation_from_state(self.game.state)
                    valid_cards = self.game.rule.get_valid_cards_from_obs(obs)
                    card_action = np.random.choice(np.flatnonzero(valid_cards))
                    self.game.action_play_card(card_action)

                score[i] = self.game.state.points[0]

            results[trump] = score

        return results
