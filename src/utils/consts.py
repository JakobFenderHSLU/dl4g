import numpy as np

_TRUMP_SUIT_VALUE = [11, 4, 3, 20, 10, 14, 0, 0, 0]
_NOT_TRUMP_SUIT_VALUE = [11, 4, 3, 2, 10, 0, 0, 0, 0]
_OBE_ABE_VALUE = [11, 4, 3, 2, 10, 0, 0, 0, 0]
_UNE_UFE_VALUE = [0, 4, 3, 2, 10, 0, 0, 0, 11]

TRUMP_VALUE_MASK = [
    # flatten [_TRUMP_SUIT_VALUE, _NOT_TRUMP_SUIT_VALUE, _NOT_TRUMP_SUIT_VALUE, _NOT_TRUMP_SUIT_VALUE]
    np.array(
        [
            _TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
        ]
    ).flatten(),
    np.array(
        [
            _NOT_TRUMP_SUIT_VALUE,
            _TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
        ]
    ).flatten(),
    np.array(
        [
            _NOT_TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
            _TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
        ]
    ).flatten(),
    np.array(
        [
            _NOT_TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
            _NOT_TRUMP_SUIT_VALUE,
            _TRUMP_SUIT_VALUE,
        ]
    ).flatten(),
    np.array(_OBE_ABE_VALUE * 4),
    np.array(_UNE_UFE_VALUE * 4),
]

_TRUMP_WEIGHT = [25, 24, 23, 27, 22, 26, 21, 20, 19]
_NOT_TRUMP_SUIT_WEIGHT = [18, 17, 16, 15, 14, 13, 12, 11, 10]
_NOT_TRUMP_WEIGHT = [9, 8, 7, 6, 5, 4, 3, 2, 1]

_OBE_ABE_WEIGHT = [18, 16, 14, 12, 10, 8, 6, 4, 2]
_OBE_ABE_SUIT_WEIGHT = [19, 17, 15, 13, 11, 9, 7, 5, 3]

_UNE_UFE_WEIGHT = [2, 4, 6, 8, 10, 12, 14, 16, 18]
_UNE_UFE_SUIT_WEIGHT = [3, 5, 7, 9, 11, 13, 15, 17, 19]

TRUMP_WEIGHTS = np.array(
    [
        # TRUMP: DIAMONDS
        [
            np.array(
                [_TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT]
            ).flatten(),  # DIAMONDS
            np.array(
                [
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                ]
            ).flatten(),  # HEARTS
            np.array(
                [
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                ]
            ).flatten(),  # SPADES
            np.array(
                [
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                ]
            ).flatten(),  # CLUBS
        ],
        # TRUMP: HEARTS
        [
            np.array(
                [
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                ]
            ).flatten(),  # DIAMONDS
            np.array(
                [_NOT_TRUMP_WEIGHT, _TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT]
            ).flatten(),  # HEARTS
            np.array(
                [
                    _NOT_TRUMP_WEIGHT,
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                ]
            ).flatten(),  # SPADES
            np.array(
                [
                    _NOT_TRUMP_WEIGHT,
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                ]
            ).flatten(),  # CLUBS
        ],
        # TRUMP: SPADES
        [
            np.array(
                [
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                ]
            ).flatten(),  # DIAMONDS
            np.array(
                [
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                ]
            ).flatten(),  # HEARTS
            np.array(
                [_NOT_TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT, _TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT]
            ).flatten(),  # SPADES
            np.array(
                [
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                ]
            ).flatten(),  # CLUBS
        ],
        # TRUMP: CLUBS
        [
            np.array(
                [
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _TRUMP_WEIGHT,
                ]
            ).flatten(),  # DIAMONDS
            np.array(
                [
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _TRUMP_WEIGHT,
                ]
            ).flatten(),  # HEARTS
            np.array(
                [
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_WEIGHT,
                    _NOT_TRUMP_SUIT_WEIGHT,
                    _TRUMP_WEIGHT,
                ]
            ).flatten(),  # SPADES
            np.array(
                [_NOT_TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT, _NOT_TRUMP_WEIGHT, _TRUMP_WEIGHT]
            ).flatten(),  # CLUBS
        ],
        # TRUMP: OBE_ABE
        [
            np.array(
                [
                    _OBE_ABE_SUIT_WEIGHT,
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_WEIGHT,
                ]
            ).flatten(),  # DIAMONDS
            np.array(
                [
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_SUIT_WEIGHT,
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_WEIGHT,
                ]
            ).flatten(),  # HEARTS
            np.array(
                [
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_SUIT_WEIGHT,
                    _OBE_ABE_WEIGHT,
                ]
            ).flatten(),  # SPADES
            np.array(
                [
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_WEIGHT,
                    _OBE_ABE_SUIT_WEIGHT,
                ]
            ).flatten(),  # CLUBS
        ],
        # TRUMP: UNE_UFE
        [
            np.array(
                [
                    _UNE_UFE_SUIT_WEIGHT,
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_WEIGHT,
                ]
            ).flatten(),  # DIAMONDS
            np.array(
                [
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_SUIT_WEIGHT,
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_WEIGHT,
                ]
            ).flatten(),  # HEARTS
            np.array(
                [
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_SUIT_WEIGHT,
                    _UNE_UFE_WEIGHT,
                ]
            ).flatten(),  # SPADES
            np.array(
                [
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_WEIGHT,
                    _UNE_UFE_SUIT_WEIGHT,
                ]
            ).flatten(),  # CLUBS
        ],
    ]
)
