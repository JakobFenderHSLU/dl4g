from unittest import TestCase

import numpy as np
from jass.game.game_observation import GameObservation

from src.trump_strategy.highest_sum_trump_strategy import HighestSumTrumpStrategy


class TestHighestSumTrumpStrategy(TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = HighestSumTrumpStrategy(log_level=self._log_level, seed=self._seed)

    def test_choose_trump_diamond(self):
        observation = GameObservation()

        hand = np.zeros(36)
        hand[0] = 1  # DIAMOND ACE
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE
        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(0, trump)

    def test_choose_trump_heart(self):
        observation = GameObservation()

        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[15] = 1  # HEART 8
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE
        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(1, trump)

    def test_choose_trump_spade(self):
        observation = GameObservation()

        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[15] = 1  # HEART 8
        hand[19] = 1  # SPADE KING
        hand[22] = 1  # SPADE 10
        hand[27] = 1  # CLUB ACE
        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(2, trump)

    def test_choose_trump_club(self):
        observation = GameObservation()

        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[15] = 1  # HEART 8
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE
        hand[31] = 1  # CLUB 10

        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(3, trump)
