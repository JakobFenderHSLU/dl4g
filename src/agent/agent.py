from typing import List

from jass.agents.agent import Agent

from play_rules import AbstractPlayRule


class CustomAgent(Agent):
    def __init__(self, trump_strategy, play_strategy, play_rules_strategies: List[AbstractPlayRule]):
        self.trump_strategy = trump_strategy
        self.play_strategy = play_strategy
        self.play_rules_strategy = play_rules_strategies

    def action_trump(self, obs):
        return self.trump_strategy.choose_trump(obs)

    def action_play_card(self, obs):
        # Check if any of the play rules strategies can play a card
        for strategy in self.play_rules_strategy:
            card = strategy.choose_card(obs)
            if card is not None:
                return card

        return self.play_strategy.choose_card(obs)
