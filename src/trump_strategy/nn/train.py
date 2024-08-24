import torch
import torch.nn as nn

from src.trump_strategy.nn.trump_selector import TrumpSelector


class Trainer:
    def __init__(self, max_epochs=10000, batch_size=64, lr=1e-3):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrumpSelector().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        print("Training...")

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
