from unittest import TestCase

import numpy as np
from jass.game.game_observation import GameObservation

from play_rule_strategy.pull_trumps_strategy import PullTrumpsPlayRuleStrategy


class TestPullTrumpsPlayRuleStrategy(TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = PullTrumpsPlayRuleStrategy(
            log_level=self._log_level, seed=self._seed
        )

    def test_choose_card_wrong_trump(self):
        obs = GameObservation()
        obs.trump = 4
        card = self._strategy.choose_card(obs)
        self.assertEqual(None, card)

        obs.trump = 5
        card = self._strategy.choose_card(obs)
        self.assertEqual(None, card)

    def test_choose_card_already_pulled(self):
        obs = GameObservation()
        obs.trump = 0
        obs.tricks = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]
        obs.trick_first_player = [0, 1, 2]
        obs.player = 0

        card = self._strategy.choose_card(obs)

        self.assertEqual(None, card)

    def test_choose_card_not_pulled(self):
        obs = GameObservation()
        obs.trump = 1
        obs.tricks = [
            [0, 1, 2, 3],
            [19, 20, 21, 22],
            [4, 5, 6, 7],
            [8, 9, 10, 17],
        ]
        obs.trick_first_player = [0, 1, 2, 3]
        obs.player = 1

        valid_cards = np.zeros(36)
        valid_cards[11] = 1
        valid_cards[12] = 1
        valid_cards[13] = 1
        valid_cards[14] = 1

        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(obs)

        self.assertEqual(12, card)

    def test_choose_card_pulled_partner(self):
        obs = GameObservation()
        obs.trump = 0
        obs.tricks = [
            [0, 1, 3, 2],
            [4, 11, 6, 10],
        ]
        obs.trick_first_player = [0, 2]
        obs.player = 0

        valid_cards = np.zeros(36)
        valid_cards[3] = 1
        valid_cards[4] = 1
        valid_cards[7] = 1
        valid_cards[9] = 1

        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(obs)

        self.assertEqual(None, card)

    def test_choose_card_pulled_current_trick(self):
        obs = GameObservation()
        obs.trump = 0
        obs.tricks = [
            [0, 1, 2, 10],
        ]
        obs.current_trick = [4, 11, 6, -1]
        obs.trick_first_player = [0]
        obs.player = 0

        valid_cards = np.zeros(36)
        valid_cards[3] = 1
        valid_cards[4] = 1
        valid_cards[7] = 1
        valid_cards[9] = 1

        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(obs)

        self.assertEqual(None, card)
