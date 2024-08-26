import numpy as np
from jass.game.const import MAX_TRUMP

from src.trump_strategy.abstract_trump_strategy import TrumpStrategy


class RandomTrumpStrategy(TrumpStrategy):
    def __init__(self, seed: int):
        super().__init__(__name__)
        self._rng = np.random.default_rng(seed)

    def choose_trump(self, observation) -> int:
        return self._rng.integers(0, MAX_TRUMP)
