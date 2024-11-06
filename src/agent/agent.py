from typing import List

from jass.agents.agent import Agent

from src.play_rule_strategy.abstract_play_rule import PlayRuleStrategy


class CustomAgent(Agent):
    def __init__(
        self,
        trump_strategy,
        play_strategy,
        play_rules_strategies: List[PlayRuleStrategy],
    ):
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

    def train(self, training_data):
        self.play_strategy.train(training_data)
