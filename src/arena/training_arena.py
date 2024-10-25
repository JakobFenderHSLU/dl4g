import numpy as np
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber

from agent.agent import CustomAgent


class TrainingArena:
    def __init__(self, players):
        self.players = players

        assert len(players) == 4, "There must be 4 players in the training arena"
        assert all(
            [isinstance(player, CustomAgent) for player in players]
        ), "All players must be of type CustomAgent"

        self.game_sim = GameSim(rule=RuleSchieber())
        self.dealing_strategy = DealingCardRandomStrategy()

    def train(self, batch_size: int = 1024, max_epochs: int = 10):
        dealer = 0  # TODO: increment dealer after each game

        for i in range(max_epochs):
            for batch in range(batch_size):
                # create game
                self.game_sim.init_from_cards(
                    self.dealing_strategy.deal_cards(), dealer
                )

                # play one round
                hands = np.zeros((batch_size, 9, 4, 36), dtype=np.int32)
                values = np.zeros((batch_size, 9, 4), dtype=np.int32)

                while not self.game_sim.is_done():
                    obs = self.game_sim.get_observation()
                    if obs.trump == -1:
                        action = self.players[obs.player].action_trump(obs)
                        self.game_sim.action_trump(action)
                    else:
                        action = self.players[obs.player].action_play_card(obs)
                        self.game_sim.action_play_card(action)

                    # TODO: beware of copy by reference
                    # TODO: beware of indexing
                    hands[batch, obs.nr_tricks, :, :] = obs.hands
                    values[batch, obs.nr_tricks, :] = obs.points[obs.player % 2]

            # for every
