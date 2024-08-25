from play_rules.abstract_play_rule import PlayRuleStrategy


class OnlyValidPlayStrategyStrategy(PlayRuleStrategy):
    def __init__(self, log_level: str, seed: int):
        super().__init__(log_level, __name__, seed)

    def choose_card(self, observation) -> int | None:
        valid_cards = self._rule.get_valid_cards_from_obs(observation)
        if sum(valid_cards) == 1:
            return int(valid_cards.argmax())
        return None
