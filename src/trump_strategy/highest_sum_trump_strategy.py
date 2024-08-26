import numpy as np
from jass.game.const import color_masks

from src.trump_strategy.abstract_trump_strategy import TrumpStrategy


class HighestSumTrumpStrategy(TrumpStrategy):
    def __init__(self):
        super().__init__(__name__)

    def choose_trump(self, observation) -> int:
        card_sum = [np.sum(observation.hand * color_mask) for color_mask in color_masks]
        chosen_trump = np.argmax(card_sum)
        self._logger.debug(f"Choosing trump: {chosen_trump} with hand {observation.hand}")
        return int(chosen_trump)
