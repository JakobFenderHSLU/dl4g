import argparse
import logging

import numpy as np
from jass.arena.arena import Arena

from src.agent.agent import CustomAgent
from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.highest_value_play_strategy import HighestValuePlayStrategy
from src.play_strategy.only_valid_play_strategy import OnlyValidPlayStrategy
from src.play_strategy.random_play_strategy import RandomPlayStrategy
from src.play_strategy.smear_play_strategy import SmearPlayStrategy
from src.trump_strategy.abstract_trump_strategy import TrumpStrategy
from src.trump_strategy.highest_score_trump_strategy import HighestScoreTrumpStrategy
from src.trump_strategy.highest_sum_trump_strategy import HighestSumTrumpStrategy
from src.trump_strategy.random_trump_strategy import RandomTrumpStrategy
from src.trump_strategy.statistical_trump_strategy import StatisticalTrumpStrategy
from src.utils.log_utils import LogUtils
from src.utils.results_utils import ResultsUtils

POSSIBLE_TRUMP_STRATEGIES = ["random", "highest_sum", "highest_score", "statistical"]
POSSIBLE_PLAY_STRATEGIES = ["random", "highest_value"]
POSSIBLE_ADDON_PLAY_STRATEGIES = ["all", "none", "only_valid", "smear"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-at", "--agent-trump-strategy",
                        default=POSSIBLE_TRUMP_STRATEGIES[0],
                        choices=POSSIBLE_TRUMP_STRATEGIES,
                        help="Choose the trump strategy for the agent")
    parser.add_argument("-ap", "--agent-play-strategy",
                        default=POSSIBLE_PLAY_STRATEGIES[0],
                        choices=POSSIBLE_PLAY_STRATEGIES,
                        help="Choose the play strategy for the agent")
    parser.add_argument("-aap", "--agent-addon-play-strategy",
                        default=POSSIBLE_ADDON_PLAY_STRATEGIES[0],
                        choices=POSSIBLE_ADDON_PLAY_STRATEGIES,
                        help="Choose the additional play strategy for the agent")
    parser.add_argument("-ot", "--opponent-trump-strategy",
                        default=POSSIBLE_TRUMP_STRATEGIES[0],
                        choices=POSSIBLE_TRUMP_STRATEGIES,
                        help="Choose the trump strategy for the opponent")
    parser.add_argument("-op", "--opponent-play-strategy",
                        default=POSSIBLE_PLAY_STRATEGIES[0],
                        choices=POSSIBLE_PLAY_STRATEGIES,
                        help="Choose the play strategy for the opponent")
    parser.add_argument("-oa", "--opponent-addon-play-strategy",
                        default=POSSIBLE_ADDON_PLAY_STRATEGIES[1],
                        choices=POSSIBLE_ADDON_PLAY_STRATEGIES,
                        help="Choose the additional play strategy for the opponent")
    parser.add_argument("-n", "--n_games", default=100, type=int, help="Number of games to play")
    parser.add_argument("-ll", "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("-s", "--seed", type=int, help="Set the seed for the random number generator")

    args = parser.parse_args()

    log_utils = LogUtils(log_level=args.log_level)
    results_utils = ResultsUtils()

    logger = logging.getLogger("run.py")

    logger.info("Running a game Simulation with the following parameters:")
    logger.info(f"Agent: {args.agent_trump_strategy} - {args.agent_play_strategy} - {args.agent_addon_play_strategy}")
    logger.info(f"Opponent: {args.opponent_trump_strategy} - {args.opponent_play_strategy} - "
                f"{args.opponent_addon_play_strategy}")
    logger.info(f"Number of games: {args.n_games}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("Starting the simulation...")


    def _get_trump_strategy(strategy_name: str) -> TrumpStrategy:
        if strategy_name == "random":
            return RandomTrumpStrategy(log_level=args.log_level, seed=args.seed)
        elif strategy_name == "highest_sum":
            return HighestSumTrumpStrategy(log_level=args.log_level, seed=args.seed)
        elif strategy_name == "highest_score":
            return HighestScoreTrumpStrategy(log_level=args.log_level, seed=args.seed)
        elif strategy_name == "statistical":
            return StatisticalTrumpStrategy(log_level=args.log_level, seed=args.seed,
                                            values_path="data/statistical/stat_values_v3.txt")
        else:
            raise ValueError(f"Unknown trump strategy: {strategy_name}")


    def _get_play_strategy(strategy_name: str, addon_strategy: str) -> PlayStrategy:
        strategy = None
        if strategy_name == "random":
            strategy = RandomPlayStrategy(log_level=args.log_level, seed=args.seed)
        elif strategy_name == "highest_value":
            strategy = HighestValuePlayStrategy(log_level=args.log_level, seed=args.seed)
        else:
            raise ValueError(f"Unknown play strategy: {strategy_name}")

        if addon_strategy == "all":
            return OnlyValidPlayStrategy(
                log_level=args.log_level,
                seed=args.seed,
                next_strategy=SmearPlayStrategy(
                    log_level=args.log_level,
                    seed=args.seed,
                    next_strategy=strategy
                )
            )
        elif addon_strategy == "none":
            return strategy
        elif addon_strategy == "only_valid":
            return OnlyValidPlayStrategy(log_level=args.log_level, seed=args.seed, next_strategy=strategy)
        elif addon_strategy == "smear":
            return SmearPlayStrategy(log_level=args.log_level, seed=args.seed, next_strategy=strategy)


    np.random.seed(args.seed)
    arena1 = Arena(nr_games_to_play=args.n_games // 2,
                   save_filename=f"logs/{log_utils.formatted_start_time}_arena_logs")
    arena1.set_players(
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy, args.agent_addon_play_strategy)),
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy, args.opponent_addon_play_strategy)),
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy, args.agent_addon_play_strategy)),
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy, args.opponent_addon_play_strategy)),
    )
    arena1.play_all_games()

    np.random.seed(args.seed)
    arena2 = Arena(nr_games_to_play=args.n_games // 2,
                   save_filename=f"logs/{log_utils.formatted_start_time}_arena_logs")
    arena2.set_players(
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy, args.opponent_addon_play_strategy)),
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy, args.agent_addon_play_strategy)),
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy, args.opponent_addon_play_strategy)),
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy, args.agent_addon_play_strategy)),
    )
    arena2.play_all_games()

    results_utils.print_results([arena1, arena2])
