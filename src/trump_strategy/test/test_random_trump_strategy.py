from unittest import TestCase

from src.trump_strategy.random_trump_strategy import RandomTrumpStrategy


class TestRandomTrumpStrategy(TestCase):
    def setUp(self):
        self._log_level = "DEBUG"
        self._seed = 42
        self._strategy = RandomTrumpStrategy(seed=self._seed)

    def test_choose_trump(self):
        trump = self._strategy.choose_trump(None)
        self.assertTrue(0 <= trump <= 3)
