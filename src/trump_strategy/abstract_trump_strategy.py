import logging
from abc import ABC, abstractmethod

from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class TrumpStrategy(ABC):

    def __init__(self, strategy_name: str):
        """
        Initialize the abstract trump strategy. This method should be implemented by the child class.
        :param strategy_name: The name of the strategy
        """
        self._strategy_name = strategy_name
        self._logger = logging.getLogger(self._strategy_name)
        self._rule = RuleSchieber()

    @abstractmethod
    def choose_trump(self, observation: GameObservation) -> int:
        """
        Choose the trump suit based on the observation. This method should be implemented by the child class.
        :param observation: The current game observation
        :return: The index of the selected card.
        """
        raise NotImplementedError("Method not implemented")
