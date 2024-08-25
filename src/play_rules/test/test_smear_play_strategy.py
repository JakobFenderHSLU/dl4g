from unittest import TestCase

import numpy as np
from jass.game.const import OBE_ABE, UNE_UFE, DIAMONDS
from jass.game.game_observation import GameObservation

from play_rules.smear_play_strategy import SmearPlayStrategy


class TestSmearPlayStrategy(TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = SmearPlayStrategy(log_level=self._log_level, seed=self._seed)

    def test_choose_card_not_last(self):
        obs = GameObservation()
        obs.current_trick = np.array([7, 1, -1, -1])

        card = self._strategy.choose_card(obs)
        self.assertEqual(card, None)

    def test_choose_card_not_save(self):
        obs = GameObservation()
        obs.trump = 0
        obs.current_trick = np.array([7, 1, 5, -1])

        card = self._strategy.choose_card(obs)
        self.assertEqual(card, None)

    def test_choose_card_smear_obe_abe(self):
        obs = GameObservation()
        obs.trump = OBE_ABE
        obs.player = 0
        valid_cards = np.zeros(36)
        valid_cards[5] = 1  # DIAMOND 9
        valid_cards[15] = 1  # HEART 8
        valid_cards[26] = 1  # SPADE 6

        obs.current_trick = np.array([7, 0, 9, -1])
        obs.nr_cards_in_trick = 3

        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(obs)
        self.assertEqual(card, 15)

    def test_choose_card_smear_une_ufe(self):
        obs = GameObservation()
        obs.trump = UNE_UFE
        obs.player = 0
        valid_cards = np.zeros(36)
        valid_cards[4] = 1  # DIAMOND 10
        valid_cards[15] = 1  # HEART 8
        valid_cards[26] = 1  # SPADE 6

        obs.current_trick = np.array([7, 8, 9, -1])
        obs.nr_cards_in_trick = 3

        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(obs)
        self.assertEqual(card, 4)

    def test_choose_card_smear_trump(self):
        obs = GameObservation()
        obs.trump = DIAMONDS
        obs.player = 0
        valid_cards = np.zeros(36)
        valid_cards[4] = 1  # DIAMOND 10
        valid_cards[15] = 1  # HEART 8
        valid_cards[26] = 1  # SPADE 6

        obs.current_trick = np.array([4, 3, 0, -1])
        obs.nr_cards_in_trick = 3

        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(obs)
        self.assertEqual(card, 4)

    def test_choose_card_smear_no_card(self):
        obs = GameObservation()
        obs.trump = UNE_UFE
        obs.player = 0
        valid_cards = np.zeros(36)
        valid_cards[5] = 1  # DIAMOND 9
        valid_cards[15] = 1  # HEART 8
        valid_cards[26] = 1  # SPADE 6

        obs.current_trick = np.array([7, 8, 9, -1])
        obs.nr_cards_in_trick = 3

        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(obs)
        self.assertEqual(card, None)
