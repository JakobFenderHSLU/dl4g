import torch
import torch.nn as nn
import wandb

from src.trump_strategy.nn.trump_selector import TrumpSelector
from trump_strategy.nn.trump_data_generator import TrumpDataGenerator


class Trainer:
    def __init__(self, data_generator: TrumpDataGenerator, batch_size: int, lr: float, weight_decay: float):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrumpSelector().to(self.device)
        self.data_generator = data_generator

        self.run = wandb.init(
            project="dl4g-trump-selection",
            name=f"bs={batch_size} lr={lr} wd={weight_decay}",
            config={
                "batch_size": self.batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )

    def train(self):
        print("Training...")

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.run.config.lr,
            weight_decay=self.run.config.weight_decay
        )
        loss_fn = nn.MSELoss()
