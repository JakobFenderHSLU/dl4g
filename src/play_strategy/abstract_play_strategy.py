import logging
from abc import ABC, abstractmethod

from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class PlayStrategy(ABC):

    def __init__(self, strategy_name: str):
        """
        Initialize the abstract trump strategy. This method should be implemented by the child class.
        :param log_level: The log level
        :param agent_name: The name of the agent
        :param seed: The seed for the random number generator
        """
        self._strategy_name = strategy_name
        self._logger = logging.getLogger(self._strategy_name)
        self._rule = RuleSchieber()

    @abstractmethod
    def choose_card(self, observation: GameObservation) -> int:
        """
        Choose the trump suit based on the observation. This method should be implemented by the child class.
        The trump suits are defined as follows:
        DIAMONDS    = 0         # Ecken / Schellen
        HEARTS      = 1         # Herz / Rosen
        SPADES      = 2         # Schaufeln / Schilten
        CLUBS       = 3         # Kreuz / Eichel
        OBE_ABE     = 4
        UNE_UFE     = 5
        :param observation: The current game observation
        :return: The selected trump suit as an integer
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def train(self, training_data):
        """
        Implement the training method for the play strategy. This method should be implemented by the child class.
        :param training_data:
        :return:
        """
        pass
