from play_strategy.abstract_play_strategy import PlayStrategy


class OnlyValidPlayStrategy(PlayStrategy):
    def __init__(self, log_level: str, seed: int, next_strategy: PlayStrategy):
        super().__init__(log_level, __name__, seed)

        self._next_strategy = next_strategy

    def choose_card(self, observation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(observation)
        if sum(valid_cards) == 1:
            return int(valid_cards.argmax())
        else:
            return self._next_strategy.choose_card(observation)
