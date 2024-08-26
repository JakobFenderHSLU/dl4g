from unittest import TestCase

import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

import src.utils.game_utils as gu


class TestGameUtils(TestCase):

    def test_is_save_trick(self):
        obs = GameObservation()
        rule = RuleSchieber()
        obs.current_trick = np.array([1, 5, 2, -1])
        obs.player = 0
        obs.trump = 0
        self.assertTrue(gu.is_safe_trick(obs, rule))

    def test_get_starting_player_of_trick(self):
        obs = GameObservation()
        obs.current_trick = np.array([0, 1, 2, -1])
        obs.player = 0
        self.assertEqual(gu.get_starting_player_of_trick(obs), 2)

        obs.current_trick = np.array([0, 1, -1, -1])
        obs.player = 0
        self.assertEqual(gu.get_starting_player_of_trick(obs), 1)

        obs.current_trick = np.array([0, -1, -1, -1])
        obs.player = 3
        self.assertEqual(gu.get_starting_player_of_trick(obs), 1)

        obs.current_trick = np.array([-1, -1, -1, -1])
        obs.player = 2
        self.assertEqual(gu.get_starting_player_of_trick(obs), 2)

    def test_validate_trump(self):
        # valid trumps
        self.assertTrue(gu.validate_trump(0))
        self.assertTrue(gu.validate_trump(1))
        self.assertTrue(gu.validate_trump(2))
        self.assertTrue(gu.validate_trump(3))
        self.assertTrue(gu.validate_trump(4))
        self.assertTrue(gu.validate_trump(5))

        # expect Assertion Error
        with self.assertRaises(AssertionError):
            gu.validate_trump(None)
        with self.assertRaises(AssertionError):
            gu.validate_trump(-1)
        with self.assertRaises(AssertionError):
            gu.validate_trump(6)

    def test_validate_player(self):
        # valid players
        self.assertTrue(gu.validate_player(0))
        self.assertTrue(gu.validate_player(1))
        self.assertTrue(gu.validate_player(2))
        self.assertTrue(gu.validate_player(3))

        # expect Assertion Error
        with self.assertRaises(AssertionError):
            gu.validate_player(None)
        with self.assertRaises(AssertionError):
            gu.validate_player(-1)
        with self.assertRaises(AssertionError):
            gu.validate_player(4)

    def test_validate_current_trick(self):
        # valid tricks
        self.assertTrue(gu.validate_current_trick(np.array([-1, -1, -1, -1])))
        self.assertTrue(gu.validate_current_trick(np.array([0, 1, 2, -1])))
        self.assertTrue(gu.validate_current_trick(np.array([35, 34, 33, -1])))
        self.assertTrue(gu.validate_current_trick(np.array([0, 1, 2, -1])))
        self.assertTrue(gu.validate_current_trick(np.array([0, 1, -1, -1])))
        self.assertTrue(gu.validate_current_trick(np.array([0, -1, -1, -1])))

        # expect Assertion Error
        with self.assertRaises(AssertionError):
            gu.validate_current_trick(None)
        with self.assertRaises(AssertionError):
            gu.validate_current_trick(np.array([0, 1, 2]))
        with self.assertRaises(AssertionError):
            gu.validate_current_trick(np.array([-1, -1, -1, -1, -1]))
        with self.assertRaises(AssertionError):
            gu.validate_current_trick(np.array([-1, -1, -1, -2]))
        with self.assertRaises(AssertionError):
            gu.validate_current_trick(np.array([-1, -1, -1, -2]))
        with self.assertRaises(AssertionError):
            gu.validate_current_trick(np.array([36, 35, 34, -1]))
        with self.assertRaises(AssertionError):
            gu.validate_current_trick(np.array([36, 35, 34, 33]))

    def test_swap_colors_sum(self):
        hand = np.zeros(36)
        hand[5] = 1  # DIAMOND 9
        hand[16] = 1  # HEART 7
        hand[17] = 1  # HEART 6
        hand[19] = 1  # SPADE KING
        hand[20] = 1  # SPADE QUEEN
        hand[21] = 1  # SPADE JACK
        hand[27] = 1  # CLUB ACE
        hand[28] = 1  # CLUB KING
        hand[29] = 1  # CLUB QUEEN
        hand[30] = 1  # CLUB JACK

        swapped_hand, color_order = gu.swap_colors(hand)
        self.assertTrue(np.array_equal(color_order, np.array([3, 2, 1, 0])))

    def test_swap_colors_tie_breaker(self):
        hand = np.zeros(36)
        hand[8] = 1  # DIAMOND 6
        hand[18] = 1  # SPADE ACE
        hand[19] = 1  # SPADE KING
        hand[20] = 1  # SPADE QUEEN
        hand[21] = 1  # SPADE JACK
        hand[27] = 1  # CLUB ACE
        hand[28] = 1  # CLUB KING
        hand[29] = 1  # CLUB QUEEN
        hand[35] = 1  # CLUB 6

        swapped_hand, color_order = gu.swap_colors(hand)
        self.assertTrue(np.array_equal(color_order, np.array([2, 3, 0, 1])))

    def test_swap_colors_very_close(self):
        hand = np.zeros(36)
        hand[8] = 1  # DIAMOND 6
        hand[18] = 1  # SPADE ACE
        hand[19] = 1  # SPADE KING
        hand[21] = 1  # SPADE JACK
        hand[23] = 1  # SPADE 9
        hand[28] = 1  # CLUB KING
        hand[29] = 1  # CLUB QUEEN
        hand[30] = 1  # CLUB JACK
        hand[32] = 1  # CLUB 9

        swapped_hand, color_order = gu.swap_colors(hand)
        self.assertTrue(np.array_equal(color_order, np.array([2, 3, 0, 1])))

    def test_swap_colors_exact(self):
        hand = np.zeros(36)
        hand[8] = 1  # DIAMOND 6
        hand[18] = 1  # SPADE ACE
        hand[19] = 1  # SPADE KING
        hand[20] = 1  # SPADE QUEEN
        hand[21] = 1  # SPADE JACK
        hand[27] = 1  # CLUB ACE
        hand[28] = 1  # CLUB KING
        hand[29] = 1  # CLUB QUEEN
        hand[30] = 1  # CLUB JACK

        swapped_hand, color_order = gu.swap_colors(hand)
        self.assertTrue(np.array_equal(color_order, np.array([3, 2, 0, 1])))

    def test_swap_colors_from_order_0(self):
        hand = np.zeros(36)
        hand[0:9] = 1  # FULL DIAMOND Hand

        swapped_hand = gu.swap_colors_from_order(hand, np.array([0, 1, 2, 3]))

        self.assertTrue(np.array_equal(swapped_hand, hand))

    def test_swap_colors_from_order_1(self):
        hand = np.zeros(36)
        hand[0:9] = 1  # FULL DIAMOND Hand

        expected_hand = np.zeros(36)
        expected_hand[9:18] = 1

        swapped_hand = gu.swap_colors_from_order(hand, np.array([1, 0, 2, 3]))

        self.assertTrue(np.array_equal(swapped_hand, expected_hand))

    def test_swap_colors_from_order_2(self):
        hand = np.zeros(36)
        hand[0:9] = 1

        expected_hand = np.zeros(36)
        expected_hand[18:27] = 1

        swapped_hand = gu.swap_colors_from_order(hand, np.array([2, 1, 0, 3]))

        self.assertTrue(np.array_equal(swapped_hand, expected_hand))

    def test_swap_colors_from_order_3(self):
        hand = np.zeros(36)
        hand[0:9] = 1

        expected_hand = np.zeros(36)
        expected_hand[27:36] = 1

        swapped_hand = gu.swap_colors_from_order(hand, np.array([3, 1, 2, 0]))

        self.assertTrue(np.array_equal(swapped_hand, expected_hand))

    def test_deck_to_onehot_hands(self):
        deck = np.arange(36)  # 0-35
        expected_hands = np.zeros((4, 36))

        expected_hands[0, 0:9] = 1
        expected_hands[1, 9:18] = 1
        expected_hands[2, 18:27] = 1
        expected_hands[3, 27:36] = 1

        hands = gu.deck_to_onehot_hands(deck)
        self.assertTrue(np.array_equal(hands, expected_hands))

    def test_get_bock_chain_suit_trump(self):
        obs = GameObservation()
        obs.player = 1
        obs.trump = 0
        obs.hand = np.zeros(36)
        obs.hand[3] = 1  # DIAMOND JACK
        obs.hand[5] = 1  # DIAMOND 9
        obs.hand[0] = 1  # DIAMOND ACE
        obs.hand[2] = 1  # DIAMOND QUEEN
        obs.hand[9] = 1  # HEART ACE
        obs.hand[12] = 1  # HEART JACK

        bock_chain = gu.get_bock_chain(obs)
        self.assertTrue(np.array_equal(bock_chain, np.array([0, 3, 5])))
