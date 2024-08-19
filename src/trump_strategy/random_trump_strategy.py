from jass.game.const import MAX_TRUMP

from src.trump_strategy.abstract_trump_strategy import TrumpStrategy


class RandomTrumpStrategy(TrumpStrategy):
    def __init__(self, log_level: str, seed: int):
        super().__init__(log_level, __name__, seed)

    def choose_trump(self, observation) -> int:
        return self._rng.integers(0, MAX_TRUMP)
