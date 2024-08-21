from unittest import TestCase

import numpy as np

import src.utils.game_utils as gu


class TestGameUtils(TestCase):
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

    def test_get_starting_player_of_trick(self):
        obs = gu.GameObservation()
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
