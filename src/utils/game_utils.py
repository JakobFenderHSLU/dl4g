import numpy as np
from jass.game.const import same_team
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule


def is_save_trick(obs: GameObservation, rule: GameRule) -> bool:
    """
    Check if the current trick is save.

    :param obs: GameObservation
    """
    validate_current_trick(obs.current_trick)
    validate_player(obs.player)

    current_winner = rule.calc_winner(obs.current_trick, get_starting_player_of_trick(obs), obs.trump)
    return bool(same_team[obs.player][current_winner])


def get_starting_player_of_trick(obs: GameObservation) -> int:
    """
    Get the id of the first player of the current trick.

    :param obs: GameObservation
    """
    validate_current_trick(obs.current_trick)
    validate_player(obs.player)

    index = np.where(obs.current_trick != -1)[0]
    if len(index) == 0:
        return obs.player
    return int((index[-1] - obs.player) % 4)


def validate_trump(trump: int) -> bool:
    """
    Check if the trump is not none and is in [0, 1, 2, 3, 4, 5].

    :param trump: int
    """
    assert trump is not None, "Trump is None"
    assert isinstance(trump, int), "Trump is not an integer"
    assert trump in [0, 1, 2, 3, 4, 5], "Trump is not in [0, 1, 2, 3, 4, 5]"

    return True


def validate_current_trick(trick: np.ndarray) -> bool:
    """
    Check if the current trick not none, is a numpy array, has length 4, contains only positive values and values less
    than 36.

    :param trick: np.ndarray
    """
    assert trick is not None, "Trick is None"
    assert isinstance(trick, np.ndarray), "Trick is not a numpy array"
    assert len(trick) == 4, "Trick length is not 4"
    assert np.all(trick >= -1), "Trick contains negative values"
    assert np.all(trick < 36), "Trick contains values greater than 35"
    assert trick[-1] == -1, "Last entry in trick is not -1"
    return True


def validate_player(player: int) -> bool:
    """
    Check if the player not none and is in [0, 1, 2, 3].

    :param player: int
    """
    assert player is not None, "Player is None"
    assert player in [0, 1, 2, 3], "Player is not in [0, 1, 2, 3]"
    return True
