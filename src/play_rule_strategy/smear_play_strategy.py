import numpy as np
from jass.game.const import OBE_ABE

import src.utils.game_utils as gu
from src.play_rule_strategy.abstract_play_rule import PlayRuleStrategy


class SmearPlayRuleStrategy(PlayRuleStrategy):
    _smear_weight = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    _obenabe_smear_weight = [0, 0, 0, 0, 2, 0, 1, 0, 0]

    def __init__(self, log_level: str, seed: int):
        super().__init__(log_level, __name__, seed)

    def choose_card(self, observation) -> int | None:
        # Skip if not last card in trick
        if observation.nr_cards_in_trick != 3:
            return None

        # Skip if the trick is not safe
        if not gu.is_safe_trick(observation, self._rule):
            return None

        # Get valid cards
        valid_cards = self._rule.get_valid_cards_from_obs(observation)

        if observation.trump == OBE_ABE:
            mask = np.tile(self._obenabe_smear_weight, 4)
        else:
            mask = np.tile(self._smear_weight, 4)

        possible_cards = valid_cards * mask

        # Skip if no card to smear
        if sum(possible_cards) == 0:
            return None

        return int(np.argmax(possible_cards))
