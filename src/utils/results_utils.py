import logging

import numpy as np
from jass.arena.arena import Arena
from prettytable import PrettyTable


class ResultsUtils:

    def __init__(self):
        self._logger = logging.getLogger("ResultsUtils")

    def print_results(self, arena: Arena):
        self._logger.info("Total Games Played: {}".format(arena.nr_games_played))

        n_team_0_wins = len(arena.points_team_0[arena.points_team_0 > arena.points_team_1])
        n_team_1_wins = len(arena.points_team_0) - n_team_0_wins

        bar_length = 50
        self._logger.info(f" {n_team_0_wins} "
                          f"{' ' * (bar_length - len(str(n_team_0_wins)) - len(str(n_team_1_wins)) - 1)}"
                          f"{n_team_1_wins}")
        self._logger.info(
            f"[{'+' * int(n_team_0_wins / arena.nr_games_played * bar_length) + '-' * int(n_team_1_wins / arena.nr_games_played * bar_length)}]")

        self._logger.info("")

        table = PrettyTable(["Overall", "Agents", "Opponents"])
        table.add_row(["Total Points", sum(arena.points_team_0), sum(arena.points_team_1)])
        table.add_row(["Average Points", sum(arena.points_team_0) / arena.nr_games_played,
                       sum(arena.points_team_1) / arena.nr_games_played])
        table.add_row(["Max Points", max(arena.points_team_0), max(arena.points_team_1)])
        table.add_row(["Min Points", min(arena.points_team_0), min(arena.points_team_1)])
        table.add_row(["Points 25th Percentile", np.quantile(arena.points_team_0, 0.25),
                       np.quantile(arena.points_team_1, 0.25)])
        table.add_row(["Points Median", np.median(arena.points_team_0), np.median(arena.points_team_1)])
        table.add_row(["Points 75th Percentile", np.quantile(arena.points_team_0, 0.75),
                       np.quantile(arena.points_team_1, 0.75)])

        [self._logger.info(" " + row) for row in table.get_string().split("\n")]

        self._logger.info("")

        # get every second row
        trump_rounds = arena.points_team_0[::2]

        table = PrettyTable(["Trump Rounds", "Agents", "Opponents"])
        table.add_row(["Total Points", sum(trump_rounds), sum(arena.points_team_1[::2])])
        table.add_row(["Average Points", sum(trump_rounds) / len(trump_rounds),
                       sum(arena.points_team_1[::2]) / len(arena.points_team_1[::2])])
        table.add_row(["Max Points", max(trump_rounds), max(arena.points_team_1[::2])])
        table.add_row(["Min Points", min(trump_rounds), min(arena.points_team_1[::2])])
        table.add_row(["Points 25th Percentile", np.quantile(trump_rounds, 0.25),
                       np.quantile(arena.points_team_1[::2], 0.25)])
        table.add_row(["Points Median", np.median(trump_rounds), np.median(arena.points_team_1[::2])])
        table.add_row(["Points 75th Percentile", np.quantile(trump_rounds, 0.75),
                       np.quantile(arena.points_team_1[::2], 0.75)])

        [self._logger.info(" " + row) for row in table.get_string().split("\n")]
