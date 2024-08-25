import logging
from abc import abstractmethod, ABC

import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class PlayRuleStrategy(ABC):
    """
    Abstract class for a play rule.
    """

    def __init__(self, log_level: str, strategy_name: str, seed: int):
        """
        Initialize the abstract play rule. This method should be implemented by the child class.
        :param log_level: The log level
        :param strategy_name: The name of the strategy
        :param seed: The seed for the random number generator
        """
        self._strategy_name = strategy_name
        self._rng = np.random.default_rng(seed)
        self._logger = logging.getLogger(self._strategy_name)
        self._logger.setLevel(log_level)
        self._rule = RuleSchieber()

    @abstractmethod
    def choose_card(self, observation: GameObservation) -> int | None:
        """
        Choose the card to play based on the observation and the playable cards.
        This method should be implemented by the child class.
        :param observation: The current game observation
        :return: The selected card as an integer
        """
        raise NotImplementedError("Method not implemented")
