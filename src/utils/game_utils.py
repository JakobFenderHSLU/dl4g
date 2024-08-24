from typing import Tuple

import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule


def is_safe_trick(obs: GameObservation, rule: GameRule) -> bool:
    """
    Check if the current trick is save.

    :param obs: GameObservation
    """
    validate_current_trick(obs.current_trick)
    validate_player(obs.player)

    current_winner = rule.calc_winner(obs.current_trick, 0, obs.trump)
    return current_winner in [1, 3]


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


def swap_colors(hand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Swap colors, to reduce the amount of possible hands by 24. See README.md for an explanation.
    :param hand: the onehot encoded hand.
    :return: The swapped hand and the mapping of the swap.
    """
    _trump_color_order = 2 ** np.array([6, 5, 4, 8, 3, 7, 2, 1, 0])

    colors = hand.reshape(4, 9)
    sum_of_colors = np.sum(colors, axis=1)
    color_score = sum_of_colors * 500 + np.sum(colors * _trump_color_order, axis=1)

    # get the order of the colors
    color_order = np.argsort(color_score)[::-1]
    hand = colors[color_order].flatten()

    return hand, color_order


def swap_colors_from_order(hand: np.ndarray, color_order: np.ndarray) -> np.ndarray:
    """
    Swap colors, to reduce the amount of possible hands by 24. See README.md for an explanation.
    :param hand: the onehot encoded hand.
    :param color_order: the order of the colors
    :return: The swapped hand.
    """
    colors = hand.reshape(4, 9)
    hand = colors[color_order].flatten()
    return hand


def deck_to_onehot_hands(deck: np.ndarray) -> np.ndarray:
    """
    Convert a deck to onehot encoded hands.

    :param deck: np.ndarray
    """
    hands = np.split(deck, 4)
    hands_onehot = np.zeros((4, 36))
    for i, hand in enumerate(hands):
        hands_onehot[i, hand] = 1

    return hands_onehot
