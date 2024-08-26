import numpy as np
from jass.game.const import PUSH

from src.trump_strategy.abstract_trump_strategy import TrumpStrategy


class StatisticalTrumpStrategy(TrumpStrategy):
    def __init__(self, values_path: str):
        super().__init__(__name__)
        self.scores = np.loadtxt(values_path)

    def choose_trump(self, observation) -> int:
        hand = observation.hand
        scores = np.sum(hand * self.scores, axis=1)

        selected_trump = np.argmax(scores)

        if selected_trump == 6:
            selected_trump = PUSH

        if observation.forehand and selected_trump == 10:
            return int(np.argmax(scores[0:5]))
        else:
            return int(selected_trump)
