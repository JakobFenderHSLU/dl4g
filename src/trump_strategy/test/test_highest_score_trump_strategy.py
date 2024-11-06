from unittest import TestCase

import numpy as np
from jass.game.const import DIAMONDS, OBE_ABE, PUSH, UNE_UFE
from jass.game.game_observation import GameObservation

from src.trump_strategy.highest_score_trump_strategy import HighestScoreTrumpStrategy


class TestHighestScoreTrumpStrategy(TestCase):

    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = HighestScoreTrumpStrategy()

    def test_choose_trump_push(self):
        observation = GameObservation()

        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[15] = 1  # HEART 8
        hand[22] = 1  # SPADE 10
        hand[27] = 1  # CLUB ACE
        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(PUSH, trump)

    def test_choose_trump_suit(self):
        observation = GameObservation()
        # _trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
        # _no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]

        hand = np.zeros(36)
        hand[0] = 1  # DIAMOND ACE (15)
        hand[2] = 1  # DIAMOND QUEEN (10)
        hand[3] = 1  # DIAMOND QUEEN (25)
        hand[7] = 1  # DIAMOND 9 (19)
        hand[17] = 1  # HEART 6 (5)
        hand[18] = 1  # SPADE ACE (9)
        hand[19] = 1  # SPADE KING (7)
        hand[27] = 1  # CLUB ACE (9)
        hand[28] = 1  # CLUB KING (7)
        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(DIAMONDS, trump)

    def test_choose_trump_obenabe(self):
        observation = GameObservation()

        hand = np.zeros(36)
        hand[0] = 1  # DIAMOND ACE (14)
        hand[2] = 1  # DIAMOND QUEEN (8)
        hand[7] = 1  # DIAMOND 7 (5)
        hand[8] = 1  # DIAMOND 6 (0)
        hand[17] = 1  # HEART 6 (0)
        hand[18] = 1  # SPADE ACE (14)
        hand[19] = 1  # SPADE KING (10)
        hand[27] = 1  # CLUB ACE (14)
        hand[28] = 1  # CLUB KING (10)
        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(OBE_ABE, trump)

    def test_choose_trump_uneufe(self):
        observation = GameObservation()

        # _uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]
        hand = np.zeros(36)
        hand[0] = 1  # DIAMOND ACE (0)
        hand[2] = 1  # DIAMOND QUEEN (0)
        hand[7] = 1  # DIAMOND 7 (9)
        hand[8] = 1  # DIAMOND 6 (11)
        hand[15] = 1  # HEART 8 (7)
        hand[16] = 1  # HEART 7 (9)
        hand[17] = 1  # HEART 6 (11)
        hand[26] = 1  # SPADE 6 (11)
        hand[35] = 1  # CLUB 6 (11)

        observation.hand = hand

        trump = self._strategy.choose_trump(observation)

        self.assertEqual(UNE_UFE, trump)
