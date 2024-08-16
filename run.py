import argparse
import logging

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

POSSIBLE_MODELS = ["random"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", default=POSSIBLE_MODELS[0], choices=POSSIBLE_MODELS,
                        help="Choose the bot")
    parser.add_argument("-o", "--opponent", default=POSSIBLE_MODELS[0], choices=["All"] + POSSIBLE_MODELS,
                        help="Choose the opponent")
    parser.add_argument("-n", "--n_games", default=100, type=int, help="Number of games to play")

    args = parser.parse_args()

    print("Running a game Simulation with the following parameters:")
    print(f"Agent: {args.agent}")
    print(f"Opponent: {args.opponent}")
    print(f"Number of games: {args.n_games}")
    print("Starting the simulation...")

    logging.basicConfig(level=logging.WARNING)

    arena = Arena(nr_games_to_play=1000, save_filename='logs/arena_games')
    player = AgentRandomSchieber()

    arena.set_players(player, player, player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))




