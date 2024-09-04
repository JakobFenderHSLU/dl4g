import math

import numpy as np
import torch
import torch.nn as nn
import wandb
from jass.game.const import MAX_TRUMP
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from trump_strategy.nn.trump_data_generator import TrumpDataGenerator
from trump_strategy.nn.trump_selector import TrumpSelector


class Trainer:
    def __init__(self, data_generator: TrumpDataGenerator, batch_size: int, max_batches: int, max_epochs: int,
                 lr: float, weight_decay: float, folds: int):

        self.data_generator = data_generator
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.max_epochs = max_epochs
        self.folds = folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrumpSelector().to(self.device)
        self.lowest_val_loss = (math.inf, 0)

        self.config = {
            "batch_size": self.batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        self.run = None

    def train(self):
        print("Starting training...")
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

        for fold, (train_index, val_index) in enumerate(KFold(n_splits=self.folds).split(dataset[0])):
            end_fold = False

            self.model = TrumpSelector().to(self.device)
            self.lowest_val_loss = (math.inf, 0)

            self.run = wandb.init(
                project="dl4g-trump-selection",
                name=f"bs={self.config['batch_size']}_lr={self.config['lr']}"
                     f"_wd={self.config['weight_decay']}_fold={fold}",
                config=self.config
            )

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.run.config.lr,
                weight_decay=self.run.config.weight_decay
            )
            loss_fn = nn.MSELoss()

            train_ds = (
                dataset[0][train_index],
                dataset[1][train_index]
            )

            val_ds = (
                dataset[0][val_index],
                dataset[1][val_index]
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
                if end_fold:
                    break

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
                            print(f"New lowest val loss: {loss.item()} at epoch {epoch}")
                            self.lowest_val_loss = (loss.item(), epoch)

                            if epoch > 1000:
                                model_path = f"data/deep_trump_strategy_epochs/trump_selector_{fold}.pt"
                                torch.save(self.model.state_dict(), model_path)
                                wandb.save(model_path)

                        if epoch - self.lowest_val_loss[1] > 500:
                            print("Early stopping")
                            end_fold = True

                        if i == 0:
                            self.run.log({
                                "val/loss": loss.item(),
                            })
