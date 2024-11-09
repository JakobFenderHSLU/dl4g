import numpy as np
from jass.game.game_observation import GameObservation

from src.play_rule_strategy.abstract_play_rule import PlayRuleStrategy
from utils.consts import _TRUMP_WEIGHT


class PullTrumpsPlayRuleStrategy(PlayRuleStrategy):
    def __init__(self, log_level: str, seed: int):
        super().__init__(log_level, "PullTrumpsPlayRuleStrategy", seed)

    def choose_card(self, obs: GameObservation) -> int | None:
        if obs.trump <= 3:
            done_pulling = [False, False]

            for first_player, trick in zip(obs.trick_first_player, obs.tricks):
                for i, opponent in enumerate([1, 3]):
                    if trick[0] // 9 == obs.trump:
                        player_offset = (opponent - first_player) % 4
                        if trick[player_offset] // 9 != obs.trump:
                            done_pulling[i] = True

            # # consider current trick
            if obs.current_trick[0] != -1:
                for player_offset in [-1, -3]:
                    player = (obs.player + player_offset) % 4
                    if obs.current_trick[0] // 9 == obs.trump:
                        if obs.current_trick[player_offset] // 9 != obs.trump:
                            i = 0 if player == 1 else 1
                            done_pulling[i] = True

            if all(done_pulling):
                return None

            # play max card of trump
            valid_cards = self._rule.get_valid_cards_from_obs(obs)
            trump_cards = valid_cards[obs.trump * 9 : obs.trump * 9 + 9] * _TRUMP_WEIGHT

            if sum(trump_cards) == 0:
                return None

            else:
                return np.argmax(trump_cards) + obs.trump * 9
