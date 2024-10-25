import numpy as np
from jass.game.const import (
    CA,
    CLUBS,
    DA,
    DIAMONDS,
    HA,
    HEARTS,
    OBE_ABE,
    PUSH,
    SA,
    SPADES,
    UNE_UFE,
)

from src.trump_strategy.abstract_trump_strategy import TrumpStrategy


class HighestScoreTrumpStrategy(TrumpStrategy):
    _trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    _no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    _obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
    _uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

    def __init__(self):
        super().__init__(__name__)

    def choose_trump(self, observation) -> int:
        scores = [self._calculate_score(observation.hand, i) for i in range(6)]
        chosen_trump = np.argmax(scores)

        if observation.forehand == -1 and scores[chosen_trump] < 68:
            return PUSH
        else:
            return int(chosen_trump)

    def _calculate_score(self, hand, trump_type) -> int:
        if trump_type == OBE_ABE:
            return np.sum(hand * np.tile(self._obenabe_score, 4))
        elif trump_type == UNE_UFE:
            return np.sum(hand * np.tile(self._uneufe_score, 4))

        value_array = np.tile(self._no_trump_score, 4)

        if trump_type == DIAMONDS:
            value_array[DA:HA] = self._trump_score
        elif trump_type == HEARTS:
            value_array[HA:SA] = self._trump_score
        elif trump_type == SPADES:
            value_array[SA:CA] = self._trump_score
        elif trump_type == CLUBS:
            value_array[CA:] = self._trump_score
        return np.sum(hand * value_array)
