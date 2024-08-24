from jass.game.game_observation import GameObservation

from src.play_strategy.abstract_play_strategy import PlayStrategy


class MockPlayStrategy(PlayStrategy):
    def __init__(self, log_level: str, seed: int, play: int):
        super().__init__(log_level, __name__, seed)
        self.play = play

    def choose_card(self, observation: GameObservation) -> int:
        return self.play
