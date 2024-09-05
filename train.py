import argparse

from trump_strategy.nn.trainer import Trainer
from trump_strategy.nn.trump_data_generator import TrumpDataGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_batches", type=int, default=960)
    parser.add_argument("--folds", type=int, default=3)

    args = parser.parse_args()

    trainer = Trainer(
        data_generator=TrumpDataGenerator(
            load_data=True,
            n_play_per_hand=20,
            backup_interval=10_000,
            max_cache_size=1_000_000
        ),
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_batches=args.max_batches,
        max_epochs=10_000_000,
        folds=args.folds
    )

    trainer.train()
