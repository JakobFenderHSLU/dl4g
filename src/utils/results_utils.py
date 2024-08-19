import logging

from jass.arena.arena import Arena


class ResultsUtils:

    def __init__(self):
        self._logger = logging.getLogger("ResultsUtils")

    def print_results(self, arena: Arena):
        self._logger.info("Total Games Played: {}".format(arena.nr_games_played))

        n_team_0_wins = len(arena.points_team_0[arena.points_team_0 > arena.points_team_1])
        n_team_1_wins = len(arena.points_team_0) - n_team_0_wins

        bar_length = 50
        win_loss_bar = "+" * int(n_team_0_wins / arena.nr_games_played * bar_length) + \
                       "-" * int(n_team_1_wins / arena.nr_games_played * bar_length)

        self._logger.info(f"Team 0 Wins: {n_team_0_wins} Team 1 Wins: {n_team_1_wins}")
        self._logger.info(f"Win Loss Draw Bar: [{win_loss_bar}]")

        self._logger.info(f"Average Points Team 0: {arena.points_team_0.mean():.2f}")
        self._logger.info(f"Average Points Team 1: {arena.points_team_1.mean():.2f}")
