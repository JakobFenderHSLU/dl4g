import logging
from typing import List

import numpy as np
from jass.arena.arena import Arena
from prettytable import PrettyTable


class ResultsUtils:

    def __init__(self):
        self._logger = logging.getLogger("ResultsUtils")

    def print_results(self, arenas: List[Arena]):
        n_games_played = sum([a.nr_games_played for a in arenas])
        self._logger.info(f"Total Games Played: {n_games_played}")

        agents_wins = len(arenas[0].points_team_0[arenas[0].points_team_0 > arenas[0].points_team_1]) + \
                      len(arenas[1].points_team_1[arenas[1].points_team_1 > arenas[1].points_team_0])
        opponents_wins = n_games_played - agents_wins

        bar_length = 50
        self._print_stacked_bar_plot(bar_length, agents_wins, opponents_wins)

        agents_points = np.concat((arenas[0].points_team_0, arenas[1].points_team_1))
        opponents_points = np.concat((arenas[0].points_team_1, arenas[1].points_team_0))
        agents_winrate = (agents_points > opponents_points).mean() * 100
        opponents_winrate = (opponents_points > agents_points).mean() * 100

        table = PrettyTable(["Overall", "Agents", "Opponents"])
        table.add_row(["Winrate", f"{agents_winrate:.2f} %", f"{opponents_winrate:.2f} %"])
        table.add_row(["Average Points", sum(agents_points) / len(agents_points),
                       sum(opponents_points) / len(opponents_points)])
        table.add_row(["Max Points", max(agents_points), max(opponents_points)])
        table.add_row(["Min Points", min(agents_points), min(opponents_points)])
        table.add_row(["Points 25th Percentile", np.quantile(agents_points, 0.25),
                       np.quantile(opponents_points, 0.25)])
        table.add_row(["Points Median", np.median(agents_points), np.median(opponents_points)])
        table.add_row(["Points 75th Percentile", np.quantile(agents_points, 0.75),
                       np.quantile(opponents_points, 0.75)])

        [self._logger.info(" " + row) for row in table.get_string().split("\n")]

        self._logger.info("")

        # get every second row
        agent_trump_rounds = np.concat((arenas[0].points_team_0[1::2], arenas[1].points_team_1[::2]))
        agent_trump_rounds_winrate = np.concat((
            arenas[0].points_team_0[1::2] > arenas[0].points_team_1[1::2],
            arenas[1].points_team_1[::2] > arenas[1].points_team_0[::2]
        )).mean() * 100
        opponent_trump_rounds = np.concat((arenas[0].points_team_1[::2], arenas[1].points_team_0[1::2]))
        opponent_trump_rounds_winrate = np.concat((
            arenas[0].points_team_1[::2] > arenas[0].points_team_0[::2],
            arenas[1].points_team_0[1::2] > arenas[1].points_team_1[1::2]
        )).mean() * 100

        table = PrettyTable(["Trump Rounds", "Agents", "Opponents"])
        table.add_row(["Winrate", f"{agent_trump_rounds_winrate:.2f} %", f"{opponent_trump_rounds_winrate:.2f} %"])
        table.add_row(["Average Points", sum(agent_trump_rounds) / len(agent_trump_rounds),
                       sum(opponent_trump_rounds) / len(opponent_trump_rounds)])
        table.add_row(["Max Points", max(agent_trump_rounds), max(opponent_trump_rounds)])
        table.add_row(["Min Points", min(agent_trump_rounds), min(opponent_trump_rounds)])
        table.add_row(["Points 25th Percentile", np.quantile(agent_trump_rounds, 0.25),
                       np.quantile(opponent_trump_rounds, 0.25)])
        table.add_row(["Points Median", np.median(agent_trump_rounds), np.median(opponent_trump_rounds)])
        table.add_row(["Points 75th Percentile", np.quantile(agent_trump_rounds, 0.75),
                       np.quantile(opponent_trump_rounds, 0.75)])

        [self._logger.info(" " + row) for row in table.get_string().split("\n")]

    def _print_stacked_bar_plot(self, bar_length, agents_wins, opponents_wins):
        #  659                                           341
        # [++++++++++++++++++++++++++++++++-----------------]
        n_games_played = agents_wins + opponents_wins
        self._logger.info(f" {agents_wins} "
                          f"{' ' * (bar_length - len(str(agents_wins)) - len(str(opponents_wins)) - 1)}"
                          f"{opponents_wins}")
        plus = '+' * int(agents_wins / n_games_played * bar_length)
        minus = '-' * int(opponents_wins / n_games_played * bar_length)
        self._logger.info(
            f"[{plus}{minus}]")
        self._logger.info("")
