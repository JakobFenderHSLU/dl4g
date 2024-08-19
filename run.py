import argparse
import logging
from datetime import datetime

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

from src.bots.random_bot import RandomBot
from src.utils.log_utils import LogUtils
from src.utils.results_utils import ResultsUtils

POSSIBLE_MODELS = ["random"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", default=POSSIBLE_MODELS[0], choices=POSSIBLE_MODELS,
                        help="Choose the bot")
    parser.add_argument("-o", "--opponent", default=POSSIBLE_MODELS[0], choices=["All"] + POSSIBLE_MODELS,
                        help="Choose the opponent")
    parser.add_argument("-n", "--n_games", default=100, type=int, help="Number of games to play")
    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Increase output verbosity")

    args = parser.parse_args()

    log_utils = LogUtils(verbose=args.verbose)
    results_utils = ResultsUtils()

    logger = logging.getLogger("run.py")

    logger.info("Running a game Simulation with the following parameters:")
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Opponent: {args.opponent}")
    logger.info(f"Number of games: {args.n_games}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info("Starting the simulation...")

    arena = Arena(nr_games_to_play=args.n_games, save_filename=f"logs/{log_utils.formatted_start_time}_arena_logs")
    player = RandomBot()

    arena.set_players(player, player, player, player)
    arena.play_all_games()

    results_utils.print_results(arena)
