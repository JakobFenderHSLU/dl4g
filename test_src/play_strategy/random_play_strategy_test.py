import unittest

import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.random_play_strategy import RandomPlayStrategy


class RandomPlayStrategyTest(unittest.TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._random_play_strategy = RandomPlayStrategy(log_level=self._log_level, seed=self._seed)

    def test_choose_card_one_option(self):
        self._random_play_strategy._rng = np.random.default_rng(42)
        observation = GameObservation()
        valid_cards = np.zeros(36)
        valid_cards[24] = 1
        self._random_play_strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards
        card = self._random_play_strategy.choose_card(observation)
        self.assertEqual(24, card)

    def test_choose_card_multiple_options(self):
        self._random_play_strategy._rng = np.random.default_rng(42)
        observation = GameObservation()
        valid_cards = np.zeros(36)
        valid_cards[5] = 1
        valid_cards[17] = 1
        valid_cards[19] = 1
        valid_cards[26] = 1
        self._random_play_strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards
        card = self._random_play_strategy.choose_card(observation)
        self.assertEqual(5, card)
