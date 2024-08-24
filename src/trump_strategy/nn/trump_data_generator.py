import logging
import os

import numpy as np
from jass.game.const import MAX_TRUMP
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber
from numpy import ndarray

import src.utils.game_utils as gu


class TrumpDataGenerator:
    def __init__(self, load_data=False, n_play_per_hand=100, backup_interval=1000, cache_size=1_000_000):
        self.cache_size = cache_size
        self.cached_decks = np.zeros((self.cache_size, 36))
        self.cached_results = np.zeros((self.cache_size, MAX_TRUMP + 1, n_play_per_hand))
        self.n_play_per_hand = n_play_per_hand
        self.backup_interval = backup_interval

        self.n_cached_results = 0
        self.n_yielded_hands = 0

        self.deck = np.arange(36)

        self.game = GameSim(RuleSchieber())
        self.game_state = GameState()
        self.game_state.dealer = 0
        self.game_state.player = 1

        self.logger = logging.getLogger(__name__)

        if load_data:
            if not os.path.exists("data/cached_decks.npy") or not os.path.exists("data/cached_results.npy"):
                self.logger.warning("no cached data found")
                return

            with open("data/cached_decks.npy", "rb") as f:
                loaded_decks = np.load(f)

            with open("data/cached_results.npy", "rb") as f:
                loaded_results = np.load(f)

            if loaded_decks.shape[0] != loaded_results.shape[0]:
                self.logger.warning("cached data is inconsistent")
                return

            self.n_cached_results = loaded_results.shape[0]

            if self.n_cached_results > self.cache_size:
                self.cached_results = np.zeros(((loaded_results // self.cache_size + 1) * self.cache_size, 36))
                self.cached_results = np.zeros((
                    (loaded_results // self.cache_size + 1) * self.cache_size, MAX_TRUMP + 1, n_play_per_hand
                ))

            self.cached_decks[:self.n_cached_results] = loaded_decks
            self.cached_results[:self.n_cached_results] = loaded_results

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_yielded_hands % self.backup_interval == 0 and self.n_yielded_hands > 0:
            self.logger.info(f"Backing up hands at {self.n_yielded_hands}...")
            self._backup_hands()

        if self.n_yielded_hands < self.n_cached_results:
            deck = self.cached_results[self.n_yielded_hands]
            results = self.cached_results[self.n_yielded_hands]
            self.n_yielded_hands += 1
            return deck, results

        # Note: No check for uniqueness needed. Because the amount of data generated to have a 0.1% chance of a
        # duplicate is ~4.33 * 10^15. So we can safely ignore this case.
        deck = self._generate_random_deck()
        results = self._get_scores(deck)

        self.cached_decks[self.n_yielded_hands] = deck
        self.cached_results[self.n_yielded_hands] = results

        self.n_yielded_hands += 1

        return deck, results

    def _backup_hands(self):
        with open("data/cached_decks.npy", "wb") as f:
            np.savetxt(f, self.cached_decks, fmt="%d")

        with open("data/cached_results.npy", "wb") as f:
            np.savetxt(f, self.cached_results, fmt="%d")

    def _generate_random_deck(self) -> ndarray:
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

        return swapped_deck

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

                score[i] = np.where(self.game.state.trick_winner == 0, self.game.state.trick_points, 0).sum()

            results[trump] = score

        return results
