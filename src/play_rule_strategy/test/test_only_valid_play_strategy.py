from unittest import TestCase

import numpy as np
from jass.game.game_observation import GameObservation

from play_rule_strategy.only_valid_play_strategy import OnlyValidPlayRuleStrategy


class TestOnlyValidPlayStrategy(TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = OnlyValidPlayRuleStrategy(
            log_level=self._log_level, seed=self._seed
        )

    def test_choose_card_one(self):
        observation = GameObservation()
        valid_cards = np.zeros(36)
        valid_cards[14] = 1
        self._strategy._rule.get_valid_cards_from_obs = lambda x: valid_cards

        card = self._strategy.choose_card(observation)

        self.assertEqual(14, card)

    def test_choose_card_many(self):
        observation = GameObservation()
        valid_cards = np.zeros(36)
        valid_cards[0] = 1
        valid_cards[15] = 1
        valid_cards[26] = 1
        observation.valid_cards = valid_cards

        card = self._strategy.choose_card(observation)

        self.assertEqual(None, card)
