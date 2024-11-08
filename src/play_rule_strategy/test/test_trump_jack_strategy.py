from unittest import TestCase

import numpy as np
from jass.game.game_observation import GameObservation

from play_rule_strategy.trump_jack_strategy import TrumpJackPlayRuleStrategy


class TestTrumpJackStrategy(TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = TrumpJackPlayRuleStrategy(
            log_level=self._log_level, seed=self._seed
        )

    def test_choose_card_no_jack(self):
        obs = GameObservation()
        obs.trump = 0
        obs.player = 0
        obs.hand = np.zeros(36)
        obs.hand[1] = 1  # DIAMOND King
        obs.hand[4] = 1  # DIAMOND 10

        card = self._strategy.choose_card(obs)

        self.assertEqual(card, None)

    def test_choose_card_opponent_nine(self):
        obs = GameObservation()
        obs.player = 0

        for suit in range(4):
            obs.trump = suit
            obs.hand = np.zeros(36)

            trump_jack = suit * 9 + 3
            trump_nine = suit * 9 + 5

            obs.hand[trump_jack] = 1
            obs.hand[suit * 9 + 4] = 1  # Trump 10

            obs.current_trick = np.array([1, trump_nine, -1, -1])
            card = self._strategy.choose_card(obs)
            self.assertEqual(card, trump_jack, "Trump Jack should be played")

            obs.current_trick = np.array([trump_nine, 6, 2, -1])
            card = self._strategy.choose_card(obs)

            self.assertEqual(card, trump_jack, "Trump Jack should be played")

    def test_choose_card_partner_nine(self):
        obs = GameObservation()
        obs.player = 0

        for suit in range(4):
            obs.trump = suit
            obs.hand = np.zeros(36)

            trump_jack = suit * 9 + 3
            trump_nine = suit * 9 + 5

            obs.hand[trump_jack] = 1
            obs.hand[suit * 9 + 4] = 1  # Trump 10

            obs.current_trick = np.array([trump_nine, 1, -1, -1])
            card = self._strategy.choose_card(obs)
            self.assertEqual(card, None, "Trump Jack should not be played")

            obs.current_trick = np.array([2, trump_nine, 8, -1])
            card = self._strategy.choose_card(obs)

            self.assertEqual(card, None, "Trump Jack should not be played")

    def test_choose_card_trump_nine_played_sub_20_points(self):
        obs = GameObservation()
        obs.player = 0
        obs.tricks = np.ones((4, 9)) * -1
        obs.tricks[0, 0] = 0
        obs.tricks[0, 1] = 7
        obs.tricks[0, 2] = 8
        obs.tricks[0, 3] = 4

        for suit in range(4):
            obs.trump = suit
            obs.current_trick = np.array(
                [suit * 9 + 0, suit * 9 + 1, suit * 9 + 2, -1]
            )  # 18 points

            trump_jack = suit * 9 + 3
            trump_nine = suit * 9 + 5

            obs.hand = np.zeros(36)

            obs.hand[trump_jack] = 1
            obs.hand[suit * 9 + 4] = 1

            # if trump_nine is already played
            obs.tricks[0, 2] = trump_nine
            card = self._strategy.choose_card(obs)
            self.assertEqual(card, None, "Trump Jack should not be played")

            # if trump_nine is in hand
            obs.tricks[0, 2] = 8
            obs.hand[trump_nine] = 1
            card = self._strategy.choose_card(obs)
            self.assertEqual(card, None, "Trump Jack should not be played")

    def test_choose_card_trump_nine_played_20_points(self):
        obs = GameObservation()
        obs.player = 0
        obs.tricks = np.ones((4, 9)) * -1
        obs.tricks[0, 0] = 0
        obs.tricks[0, 1] = 7
        obs.tricks[0, 2] = 8
        obs.tricks[0, 3] = 4

        for suit in range(4):
            obs.trump = suit
            obs.current_trick = np.array([suit * 9, 13, -1, -1])  # 21 points

            trump_jack = suit * 9 + 3
            trump_nine = suit * 9 + 5

            obs.hand = np.zeros(36)

            obs.hand[trump_jack] = 1
            obs.hand[suit * 9 + 4] = 1

            # if trump_nine is already played
            obs.tricks[0, 2] = trump_nine
            card = self._strategy.choose_card(obs)
            self.assertEqual(card, trump_jack, "Trump Jack should be played")

            # if trump_nine is in hand
            obs.tricks[0, 2] = 8
            obs.hand[trump_nine] = 1
            card = self._strategy.choose_card(obs)
            self.assertEqual(card, trump_jack, "Trump Jack should be played")

            # if trick is safe
            obs.current_trick = np.array([7, suit * 9, 13, -1])  # 21 points
            card = self._strategy.choose_card(obs)
            self.assertEqual(card, None, "Trump Jack should not be played")
