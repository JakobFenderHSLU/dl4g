import numpy as np
from jass.game.game_observation import GameObservation

from src.play_strategy.abstract_play_strategy import PlayStrategy


class RandomPlayStrategy(PlayStrategy):
    def __init__(self, seed: int = None):
        super().__init__(__name__)
        self._rng = np.random.default_rng(seed)

    def choose_card(self, observation: GameObservation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(observation)
        return self._rng.choice(np.flatnonzero(valid_cards))
