import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.abstract_play_strategy import PlayStrategy
from utils.consts import TRUMP_WEIGHTS


class HighestValuePlayStrategy(PlayStrategy):
    # leave one empty for the cards for current suit cards
    _trump_weight = [25, 24, 23, 27, 22, 26, 21, 20, 19]
    _no_trump_weight = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    _no_trump_weight_current_suit = [18, 17, 16, 15, 14, 13, 12, 11, 10]

    _obenabe_weight = [18, 16, 14, 12, 10, 8, 6, 4, 2]
    _obenabe_weight_current_suit = [19, 17, 15, 13, 11, 9, 7, 5, 3]

    _uneufe_weight = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    _uneufe_weight_current_suit = [3, 5, 7, 9, 11, 13, 15, 17, 19]

    def __init__(self):
        super().__init__(__name__)

    def choose_card(self, observation: GameObservation) -> int:
        current_trump = observation.trump
        current_suit = observation.current_trick[0] // 9
        possible_cards = self._rule.get_valid_cards_from_obs(observation)

        return int(
            np.argmax(possible_cards * TRUMP_WEIGHTS[current_trump][current_suit])
        )
