import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.abstract_play_strategy import PlayStrategy


class RandomPlayStrategy(PlayStrategy):
    def __init__(self, log_level: str, seed: int):
        super().__init__(log_level, __name__, seed)

    def choose_card(self, observation: GameObservation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(observation)
        return self._rng.choice(np.flatnonzero(valid_cards))
