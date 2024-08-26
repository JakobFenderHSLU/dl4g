import math

import numpy as np
import torch
import torch.nn as nn
import wandb
from jass.game.const import MAX_TRUMP
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.trump_strategy.nn.trump_selector import TrumpSelector
from trump_strategy.nn.trump_data_generator import TrumpDataGenerator


class Trainer:
    def __init__(self, data_generator: TrumpDataGenerator, batch_size: int, max_batches: int, max_epochs: int,
                 lr: float, weight_decay: float):

        self.data_generator = data_generator
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.max_epochs = max_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrumpSelector().to(self.device)
        self.lowest_val_loss = (math.inf, 0)

        self.run = wandb.init(
            project="dl4g-trump-selection",
            name=f"bs={batch_size} lr={lr} wd={weight_decay}",
            config={
                "batch_size": self.batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
            },
        )

    def train(self):
        print("Starting training...")

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.run.config.lr,
            weight_decay=self.run.config.weight_decay
        )
        loss_fn = nn.MSELoss()

        dataset = (
            np.zeros((self.max_batches, self.batch_size, 36)),
            np.zeros((self.max_batches, self.batch_size, MAX_TRUMP + 1))
        )

        print(f"Generating {self.batch_size} hands...")
        for i in tqdm(range(self.max_batches)):
            for j in range(self.batch_size):
                hand, scores = next(self.data_generator)
                dataset[0][i][j] = hand[0]
                dataset[1][i][j] = scores.mean(axis=1)

        print(f"Generated {self.batch_size} hands.")

        train_ds = (
            dataset[0][:int(self.max_batches * 0.8)],
            dataset[1][:int(self.max_batches * 0.8)]
        )

        val_ds = (
            dataset[0][int(self.max_batches * 0.8):],
            dataset[1][int(self.max_batches * 0.8):]
        )

        train_ds = TensorDataset(
            torch.tensor(train_ds[0], dtype=torch.float).to(self.device),
            torch.tensor(train_ds[1], dtype=torch.float).to(self.device)
        )

        val_ds = TensorDataset(
            torch.tensor(val_ds[0], dtype=torch.float).to(self.device),
            torch.tensor(val_ds[1], dtype=torch.float).to(self.device)
        )

        train_dl = DataLoader(train_ds, batch_size=self.batch_size)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size)

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}")
            self.run.log({"train/epoch": epoch})
            for i, (hand, score) in enumerate(train_dl):
                optimizer.zero_grad()
                y_pred = self.model(hand)
                loss = loss_fn(y_pred, score)
                loss.backward()
                optimizer.step()

                if i == 0:
                    self.run.log({
                        "train/loss": loss.item(),
                    })

            with torch.no_grad():
                for i, (hand, score) in enumerate(val_dl):
                    y_pred = self.model(hand)
                    loss = loss_fn(y_pred, score)

                    if loss.item() < self.lowest_val_loss[0]:
                        self.lowest_val_loss = (loss.item(), epoch)

                    if self.lowest_val_loss[1] - epoch > 50:
                        print("Early stopping")
                        return

                    if i == 0:
                        self.run.log({
                            "val/loss": loss.item(),
                        })

            if epoch % 10 == 0:
                model_path = f"data/deep_trump_strategy_epochs/trump_selector_{epoch}.pt"
                torch.save(self.model.state_dict(), model_path)
                wandb.save(model_path)
