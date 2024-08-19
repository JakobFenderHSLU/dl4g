from jass.agents.agent import Agent


class CustomAgent(Agent):
    def __init__(self, trump_strategy, play_strategy):
        self.trump_strategy = trump_strategy
        self.play_strategy = play_strategy

    def action_trump(self, obs):
        return self.trump_strategy.choose_trump(obs)

    def action_play_card(self, obs):
        return self.play_strategy.choose_card(obs)
