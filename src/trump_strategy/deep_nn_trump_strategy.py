import torch
from jass.game.const import PUSH
from jass.game.game_observation import GameObservation

from src.trump_strategy.abstract_trump_strategy import TrumpStrategy
from src.trump_strategy.nn.trump_selector import TrumpSelector


class DeepNNTrumpStrategy(TrumpStrategy):
    def __init__(self, model_path: str = "data/deep_nn_trump_selector.pt"):
        super().__init__(__name__)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(model_path, weights_only=True, map_location=device)

        self.model = TrumpSelector()
        self.model.load_state_dict(weights)

    def choose_trump(self, observation: GameObservation) -> int:
        hand = observation.hand
        hand = torch.tensor(hand, dtype=torch.float32).unsqueeze(0)
        expected_points = self.model(hand)[0]
        best_trump = expected_points.argmax().item()
        if observation.forehand == -1 and expected_points[best_trump] < 69:
            return PUSH
        return best_trump
