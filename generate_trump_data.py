import argparse
import logging

import numpy as np
import pandas as pd
from jass.game.const import MAX_TRUMP
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber
from tqdm import tqdm

from utils.log_utils import LogUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hands", default=1000, type=int, help="Number of hands to play")
    parser.add_argument("--n-play-per-hand",
                        default=100, type=int, help="Number of games to play per hand")
    parser.add_argument("-ll", "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("-s", "--seed", type=int, help="Set the seed for the random number generator")

    args = parser.parse_args()

    log_utils = LogUtils(log_level=args.log_level)
    logger = logging.getLogger("generate_trump_data.py")

    logger.info("Generating:")
    logger.info(f"Number of games to play per hand: {args.n_play_per_hand}")

    logger.info("Starting the simulation...")

    np.random.seed(args.seed)

    # generate args.n_hands unique hands
    unique_shuffeled_decks = []
    deck = np.arange(36)  # 0-35

    logger.info(f"Generating {args.n_hands} unique hands...")
    for i_hand in tqdm(range(args.n_hands)):
        # shuffle cards
        np.random.shuffle(deck)

        # repeat until unique
        while np.any([np.array_equal(deck, hand) for hand in unique_shuffeled_decks]):
            np.random.shuffle(deck)

        unique_shuffeled_decks.append(deck.copy())

    onehot_encoded_hands = []

    for deck in unique_shuffeled_decks:
        hands = np.split(deck, 4)
        hands_onehot = np.zeros((4, 36))
        for i, hand in enumerate(hands):
            hands_onehot[i, hand] = 1

        onehot_encoded_hands.append(hands_onehot)

    logger.info(f"Len of onehot_encoded_hands: {len(onehot_encoded_hands)}")

    game = GameSim(RuleSchieber())
    game_state = GameState()
    game_state.dealer = 0
    game_state.player = 1

    results = []

    for hand in tqdm(onehot_encoded_hands):
        game_state.hands = hand

        game.init_from_state(game_state)

        for trump in range(MAX_TRUMP + 1):
            score = []
            np.random.seed(args.seed)
            for _ in range(args.n_play_per_hand):
                game_state.trump = trump
                game.init_from_state(game_state)

                game.action_trump(trump)

                for cards in range(36):
                    obs = observation_from_state(game.state)
                    valid_cards = game.rule.get_valid_cards_from_obs(obs)
                    card_action = np.random.choice(np.flatnonzero(valid_cards))
                    game.action_play_card(card_action)

                score.append(int(game.state.points[0]))

            results.append((hand[0].astype(int).tolist(), trump, score))

    df = pd.DataFrame(results, columns=["hand", "trump", "score"]).to_csv("data/trump_data.csv", index=False)
