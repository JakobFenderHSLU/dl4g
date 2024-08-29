import argparse
import logging
from typing import List

import numpy as np
from jass.arena.arena import Arena

from src.agent.agent import CustomAgent
from src.play_rule_strategy.abstract_play_rule import PlayRuleStrategy
from src.play_rule_strategy.only_valid_play_strategy import OnlyValidPlayRuleStrategy
from src.play_rule_strategy.smear_play_strategy import SmearPlayRuleStrategy
from src.play_strategy.abstract_play_strategy import PlayStrategy
from src.play_strategy.highest_value_play_strategy import HighestValuePlayStrategy
from src.play_strategy.information_set_mcts_play_stategy import InformationSetMCTSPlayStrategy
from src.play_strategy.random_play_strategy import RandomPlayStrategy
from src.trump_strategy.abstract_trump_strategy import TrumpStrategy
from src.trump_strategy.deep_nn_trump_strategy import DeepNNTrumpStrategy
from src.trump_strategy.highest_score_trump_strategy import HighestScoreTrumpStrategy
from src.trump_strategy.highest_sum_trump_strategy import HighestSumTrumpStrategy
from src.trump_strategy.random_trump_strategy import RandomTrumpStrategy
from src.trump_strategy.statistical_trump_strategy import StatisticalTrumpStrategy
from src.utils.log_utils import LogUtils
from src.utils.results_utils import ResultsUtils

POSSIBLE_TRUMP_STRATEGIES = ["random", "highest_sum", "highest_score", "statistical", "deep_nn"]
POSSIBLE_PLAY_STRATEGIES = ["random", "highest_value", "mcts"]
POSSIBLE_PLAY_RULE_STRATEGIES = ["all", "none", "only_valid", "smear"]

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
    parser.add_argument("-apr", "--agent-play-rule-strategies",
                        default=POSSIBLE_PLAY_RULE_STRATEGIES[1],
                        choices=POSSIBLE_PLAY_RULE_STRATEGIES,
                        nargs="+",
                        help="Choose the additional play strategy for the agent")
    parser.add_argument("-ot", "--opponent-trump-strategy",
                        default=POSSIBLE_TRUMP_STRATEGIES[0],
                        choices=POSSIBLE_TRUMP_STRATEGIES,
                        help="Choose the trump strategy for the opponent")
    parser.add_argument("-op", "--opponent-play-strategy",
                        default=POSSIBLE_PLAY_STRATEGIES[0],
                        choices=POSSIBLE_PLAY_STRATEGIES,
                        help="Choose the play strategy for the opponent")
    parser.add_argument("-oa", "--opponent-play-rule-strategies",
                        default=POSSIBLE_PLAY_RULE_STRATEGIES[1],
                        choices=POSSIBLE_PLAY_RULE_STRATEGIES,
                        nargs="+",
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
    logger.info(f"Agent: {args.agent_trump_strategy} - {args.agent_play_strategy} - {args.agent_play_rule_strategies}")
    logger.info(f"Opponent: {args.opponent_trump_strategy} - {args.opponent_play_strategy} - "
                f"{args.opponent_play_rule_strategies}")
    logger.info(f"Number of games: {args.n_games}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("Starting the simulation...")


    def _get_trump_strategy(strategy_name: str) -> TrumpStrategy:
        if strategy_name == "random":
            return RandomTrumpStrategy(seed=args.seed)
        elif strategy_name == "highest_sum":
            return HighestSumTrumpStrategy()
        elif strategy_name == "highest_score":
            return HighestScoreTrumpStrategy()
        elif strategy_name == "statistical":
            return StatisticalTrumpStrategy(values_path="data/statistical/stat_values_v3.txt")
        elif strategy_name == "deep_nn":
            return DeepNNTrumpStrategy()
        else:
            raise ValueError(f"Unknown trump strategy: {strategy_name}")


    def _get_play_strategy(strategy_name: str) -> PlayStrategy:
        if strategy_name == "random":
            return RandomPlayStrategy(seed=args.seed)
        elif strategy_name == "highest_value":
            return HighestValuePlayStrategy()
        elif strategy_name == "mcts":
            return InformationSetMCTSPlayStrategy()
        else:
            raise ValueError(f"Unknown play strategy: {strategy_name}")


    def _get_play_rule_strategies(strategies_names: str) -> List[PlayRuleStrategy]:
        if strategies_names == "none":
            return []

        if strategies_names == "all":
            strategies_names = POSSIBLE_PLAY_RULE_STRATEGIES[1:]

        strategies = []
        for strategy_name in strategies_names:
            if strategy_name == "only_valid":
                strategies.append(OnlyValidPlayRuleStrategy(seed=args.seed))
            if strategy_name == "smear":
                strategies.append(SmearPlayRuleStrategy(seed=args.seed))

        return strategies


    np.random.seed(args.seed)
    arena1 = Arena(nr_games_to_play=args.n_games // 2,
                   save_filename=f"logs/{log_utils.formatted_start_time}_arena_logs")
    arena1.set_players(
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy),
                    _get_play_rule_strategies(args.agent_play_rule_strategies)),
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy),
                    _get_play_rule_strategies(args.opponent_play_rule_strategies)),
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy),
                    _get_play_rule_strategies(args.agent_play_rule_strategies)),
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy),
                    _get_play_rule_strategies(args.opponent_play_rule_strategies))
    )
    arena1.play_all_games()

    np.random.seed(args.seed)
    arena2 = Arena(nr_games_to_play=args.n_games // 2,
                   save_filename=f"logs/{log_utils.formatted_start_time}_arena_logs")
    arena2.set_players(
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy),
                    _get_play_rule_strategies(args.opponent_play_rule_strategies)),
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy),
                    _get_play_rule_strategies(args.agent_play_rule_strategies)),
        CustomAgent(_get_trump_strategy(args.opponent_trump_strategy),
                    _get_play_strategy(args.opponent_play_strategy),
                    _get_play_rule_strategies(args.opponent_play_rule_strategies)),
        CustomAgent(_get_trump_strategy(args.agent_trump_strategy),
                    _get_play_strategy(args.agent_play_strategy),
                    _get_play_rule_strategies(args.agent_play_rule_strategies))
    )
    arena2.play_all_games()

    results_utils.print_results([arena1, arena2])
