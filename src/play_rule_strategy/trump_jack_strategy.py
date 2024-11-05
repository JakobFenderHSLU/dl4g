import numpy as np
from jass.game.game_observation import GameObservation

from play_rule_strategy.abstract_play_rule import PlayRuleStrategy
from utils.game_utils import is_safe_trick


class TrumpJackPlayRuleStrategy(PlayRuleStrategy):
    _trump_suit_value = [11, 4, 3, 20, 10, 14, 0, 0, 0]
    _no_trump_suit_value = [11, 4, 3, 2, 10, 0, 0, 0, 0]

    _trump_value_mask = np.tile(_no_trump_suit_value * 4, (6, 1))

    _trump_value_mask[4] = [11, 3, 4, 2, 10, 0, 8, 0, 0] * 4
    _trump_value_mask[5] = [0, 3, 4, 2, 10, 0, 8, 0, 11] * 4

    for i in range(4):
        _trump_value_mask[i, i * 9:i * 9 + 9] = _trump_suit_value

    def __init__(self, log_level: str, seed: int):
        super().__init__(log_level, "TrumpJackPlayRuleStrategy", seed)

    def choose_card(self, obs: GameObservation) -> int | None:
        # Skip if trump is not a suit
        if obs.trump >= 4:
            return None

        # check if the player has the trump jack
        trump_jack = obs.trump * 9 + 3
        trump_nine = obs.trump * 9 + 5
        if obs.hand[trump_jack] != 1:
            return None

        if obs.hand[trump_nine] != 1:
            # if trump 9 in current trick by enemy player, play trump jack
            current_index = np.where(obs.current_trick == -1)[0][0]
            if obs.current_trick[current_index - 1] == trump_nine or obs.current_trick[current_index - 3] == trump_nine:
                return trump_jack

        # if trump has been played, or we have it, and we won't win the trick by default
        # and the current trick is worth more than 20 points, play the trump jack
        if ((np.any(trump_nine in obs.tricks) or obs.hand[trump_nine] == 1) and not is_safe_trick(obs)
                and np.sum(self._trump_value_mask[obs.trump][obs.current_trick]) > 20):
            return trump_jack
