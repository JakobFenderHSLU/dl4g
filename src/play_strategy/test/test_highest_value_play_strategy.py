from unittest import TestCase

import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.highest_value_play_strategy import HighestValuePlayStrategy


class TestHighestValuePlayStrategy(TestCase):

    def setUp(self):
        self._log_level = "DEBUG"
        self._strategy = HighestValuePlayStrategy()

    def test_choose_suit_first(self):
        observation = GameObservation()

        observation.trump = 3  # CLUB
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [-1, -1, -1, -1]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(27, card)

    def test_choose_suit_highest(self):
        observation = GameObservation()

        observation.trump = 3  # CLUB
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [7, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(27, card)

    def test_choose_suit_normal(self):
        observation = GameObservation()

        observation.trump = 1  # HEART
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [7, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(17, card)

    def test_choose_suit_tie(self):
        observation = GameObservation()

        observation.trump = 1  # DIAMOND
        hand = np.zeros(36)
        hand[10] = 1  # HEART KING
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [11, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(10, card)

    def test_choose_obenabe(self):
        observation = GameObservation()

        observation.trump = 4  # OBE_ABE
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [22, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(27, card)

    def test_choose_obenabe_suit(self):
        observation = GameObservation()

        observation.trump = 4  # OBE_ABE
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[18] = 1  # SPADE ACE
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [22, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(18, card)

    def test_choose_obenabe_tie(self):
        observation = GameObservation()

        observation.trump = 4  # OBE_ABE
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[18] = 1  # SPADE ACE
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [3, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(18, card)

    def test_choose_suit_uneufe(self):
        observation = GameObservation()

        observation.trump = 5  # UNE_UFE
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [22, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(17, card)

    def test_choose_suit_uneufe_suit(self):
        observation = GameObservation()

        observation.trump = 5  # UNE_UFE
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[26] = 1  # SPADE 6
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [22, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(26, card)

    def test_choose_suit_uneufe_tie(self):
        observation = GameObservation()

        observation.trump = 5  # UNE_UFE
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[26] = 1  # SPADE 6
        hand[27] = 1  # CLUB ACE

        observation.hand = hand
        observation.current_trick = [3, 0, 0, 0]

        # ALL CARDS ARE VALID
        self._strategy._rule.get_valid_cards_from_obs = lambda x: hand
        card = self._strategy.choose_card(observation)
        self.assertEqual(17, card)
