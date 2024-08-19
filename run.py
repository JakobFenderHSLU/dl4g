import argparse
import logging

from jass.arena.arena import Arena

from src.agent.agent import CustomAgent
from src.play_strategy.random_play_strategy import RandomPlayStrategy
from src.trump_strategy.random_trump_strategy import RandomTrumpStrategy
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
    parser.add_argument("-ll", "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("-s", "--seed", default=42, type=int, help="Set the seed for the random number generator")

    args = parser.parse_args()

    log_utils = LogUtils(log_level=args.log_level)
    results_utils = ResultsUtils()

    logger = logging.getLogger("run.py")

    logger.info("Running a game Simulation with the following parameters:")
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Opponent: {args.opponent}")
    logger.info(f"Number of games: {args.n_games}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("Starting the simulation...")

    if args.agent == "random":
        trump_strategy = RandomTrumpStrategy(log_level=args.log_level, seed=args.seed)
        play_strategy = RandomPlayStrategy(log_level=args.log_level, seed=args.seed)
        agent_team = [CustomAgent(trump_strategy, play_strategy), CustomAgent(trump_strategy, play_strategy)]

    if args.opponent == "random":
        trump_strategy = RandomTrumpStrategy(log_level=args.log_level, seed=args.seed)
        play_strategy = RandomPlayStrategy(log_level=args.log_level, seed=args.seed)
        opponent_team = [CustomAgent(trump_strategy, play_strategy), CustomAgent(trump_strategy, play_strategy)]

    arena = Arena(nr_games_to_play=args.n_games, save_filename=f"logs/{log_utils.formatted_start_time}_arena_logs")

    arena.set_players(agent_team[0], opponent_team[0], agent_team[1], opponent_team[1])
    arena.play_all_games()

    results_utils.print_results(arena)
