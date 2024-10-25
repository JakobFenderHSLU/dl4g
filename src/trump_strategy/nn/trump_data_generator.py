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
    def __init__(
        self,
        load_data=False,
        n_play_per_hand=100,
        backup_interval=10_000,
        max_cache_size=1_000_000,
    ):
        self.DATA_PATH = "data/trump_data_generator"

        self.max_cache_size = max_cache_size
        self.n_play_per_hand = n_play_per_hand
        self.backup_interval = backup_interval

        self.total_n_cached_results = 0
        self.relative_n_cached_results = 0
        self.total_n_yielded_hands = 0
        self.relative_n_yielded_hands = 0

        self.deck = np.arange(36)

        self.game = GameSim(RuleSchieber())
        self.game_state = GameState()
        self.game_state.dealer = 0
        self.game_state.player = 1

        self.logger = logging.getLogger(__name__)

        if load_data:
            self._load_data()
        else:
            self.cached_decks = np.zeros((self.max_cache_size, 36)).astype(int)
            self.cached_results = np.zeros(
                (self.max_cache_size, MAX_TRUMP + 1, n_play_per_hand)
            ).astype(int)

    def __iter__(self):
        return self

    def __next__(self):
        if self.total_n_yielded_hands >= self.max_cache_size:
            self._load_data()
            self.relative_n_yielded_hands = 0

        # If we have cached data, return it
        if self.total_n_yielded_hands < self.total_n_cached_results:
            deck = self.cached_decks[self.relative_n_yielded_hands]
            onehot_hands = gu.deck_to_onehot_hands(deck)
            results = self.cached_results[self.relative_n_yielded_hands]

            self.total_n_yielded_hands += 1
            self.relative_n_yielded_hands += 1

            return onehot_hands, results

        # Backup data at interval if it was generated.
        elif (
            self.total_n_yielded_hands % self.backup_interval == 0
            and self.total_n_yielded_hands > 0
        ):
            self.logger.info(f"Backing up hands at {self.total_n_yielded_hands}...")
            self._backup_hands()

        # Note: No check for uniqueness needed. Because the amount of data generated to have a 0.1% chance of a
        # duplicate is ~4.33 * 10^15. So we can safely ignore this case.
        onehot_hands = self._generate_random_deck()
        results = self._get_scores(onehot_hands)

        self.cached_decks[self.relative_n_yielded_hands] = np.where(onehot_hands == 1)[
            1
        ]
        self.cached_results[self.relative_n_yielded_hands] = results

        self.total_n_yielded_hands += 1
        self.relative_n_yielded_hands += 1

        return onehot_hands, results

    def _load_data(self):
        cache_to_load = self.total_n_yielded_hands // self.max_cache_size
        if not os.path.exists(
            f"{self.DATA_PATH}/cached_decks_{cache_to_load}.npy"
        ) or not os.path.exists(f"{self.DATA_PATH}/cached_results_{cache_to_load}.npy"):
            self.logger.warning("no cached data found")
            return

        # load as int
        loaded_decks = np.load(
            f"{self.DATA_PATH}/cached_decks_{cache_to_load}.npy"
        ).astype(int)
        loaded_results = np.load(
            f"{self.DATA_PATH}/cached_results_{cache_to_load}.npy"
        ).astype(int)

        if loaded_decks is None or loaded_results is None:
            self.logger.warning("cached data is None")
            self.cache_decks = np.zeros((self.max_cache_size, 36)).astype(int)
            self.cached_results = np.zeros(
                (self.max_cache_size, MAX_TRUMP + 1, self.n_play_per_hand)
            ).astype(int)

        if loaded_decks.shape[0] != loaded_results.shape[0]:
            self.logger.warning("cached data is inconsistent")
            self.cache_decks = np.zeros((self.max_cache_size, 36)).astype(int)
            self.cached_results = np.zeros(
                (self.max_cache_size, MAX_TRUMP + 1, self.n_play_per_hand)
            ).astype(int)

        first_empty_hand_index = np.where(loaded_decks.sum(axis=1) == 0)[0][0]
        self.total_n_cached_results += first_empty_hand_index
        self.relative_n_cached_results = first_empty_hand_index

        self.cached_decks = loaded_decks
        self.cached_results = loaded_results

    def _backup_hands(self):
        with open(
            f"{self.DATA_PATH}/cached_decks_{self.total_n_yielded_hands // self.max_cache_size}.npy",
            "wb",
        ) as f:
            np.save(f, self.cached_decks.astype(int))

        with open(
            f"{self.DATA_PATH}/cached_results_{self.total_n_yielded_hands // self.max_cache_size}.npy",
            "wb",
        ) as f:
            np.save(f, self.cached_results.astype(int))

    def _generate_random_deck(self) -> ndarray:
        deck = self.deck.copy()
        np.random.shuffle(deck)
        hands = gu.deck_to_onehot_hands(deck)
        swap_hand, color_order = gu.swap_colors(hands[0])
        swapped_hands = np.array(
            [
                swap_hand,
                gu.swap_colors_from_order(hands[1], color_order),
                gu.swap_colors_from_order(hands[2], color_order),
                gu.swap_colors_from_order(hands[3], color_order),
            ]
        )

        return swapped_hands

    def _get_scores(self, hands: np.ndarray) -> ndarray:
        self.game_state.hands = hands

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

                score[i] = np.where(
                    self.game.state.trick_winner == 0, self.game.state.trick_points, 0
                ).sum()

            results[trump] = score

        return results
