import argparse
import logging

import numpy as np
from tqdm import tqdm

from src.trump_strategy.nn.trump_data_generator import TrumpDataGenerator
from src.utils.log_utils import LogUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-hands", default=1001, type=int, help="Number of hands to play"
    )
    parser.add_argument(
        "--n-play-per-hand",
        default=20,
        type=int,
        help="Number of games to play per hand",
    )
    parser.add_argument(
        "--backup-interval",
        default=10_000,
        type=int,
        help="Interval to backup the data",
    )
    parser.add_argument(
        "--max-cache-size", default=1_000_000, type=int, help="Size of the cache"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="Set the seed for the random number generator"
    )
    parser.add_argument("--load-data", action="store_true", help="Load cached data")

    args = parser.parse_args()

    log_utils = LogUtils(log_level=args.log_level)
    logger = logging.getLogger("generate_trump_data.py")

    logger.info("Generating:")
    logger.info(f"Number of games to play per hand: {args.n_play_per_hand}")

    logger.info("Starting the simulation...")

    np.random.seed(args.seed)

    trump_data_generator = TrumpDataGenerator(
        load_data=args.load_data,
        n_play_per_hand=args.n_play_per_hand,
        backup_interval=args.backup_interval,
        max_cache_size=args.max_cache_size,
    )

    for _ in tqdm(range(args.n_hands)):
        next(trump_data_generator)

    logger.info("Finished the simulation")
