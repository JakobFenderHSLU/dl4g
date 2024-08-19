import logging
from abc import ABC, abstractmethod

import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class TrumpStrategy(ABC):

    def __init__(self, log_level: str, strategy_name: str, seed: int):
        """
        Initialize the abstract trump strategy. This method should be implemented by the child class.
        :param log_level: The log level
        :param agent_name: The name of the agent
        :param seed: The seed for the random number generator
        """
        self._strategy_name = strategy_name
        self._rng = np.random.default_rng(seed)
        self._logger = logging.getLogger(self._strategy_name)
        self._logger.setLevel(log_level)
        self._rule = RuleSchieber()

    @abstractmethod
    def choose_trump(self, observation: GameObservation) -> int:
        """
        Choose the trump suit based on the observation. This method should be implemented by the child class.
        :param observation: The current game observation
        :return: The index of the selected card.
        """
        raise NotImplementedError("Method not implemented")
