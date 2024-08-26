from unittest import TestCase

import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.random_play_strategy import RandomPlayStrategy


class TestRandomPlayStrategy(TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = RandomPlayStrategy(seed=self._seed)

    def test_choose_card_one_option(self):
        observation = GameObservation()
        valid_cards = np.zeros(36)
        valid_cards[24] = 1
        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards
        card = self._strategy.choose_card(observation)
        self.assertEqual(24, card)

    def test_choose_card_multiple_options(self):
        self._strategy._rng = np.random.default_rng(42)
        observation = GameObservation()
        valid_cards = np.zeros(36)
        valid_cards[5] = 1  # DIAMOND 9
        valid_cards[17] = 1  # HEART 6
        valid_cards[19] = 1  # SPADE KING
        valid_cards[27] = 1  # CLUB ACE
        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards
        card = self._strategy.choose_card(observation)
        self.assertEqual(5, card)
