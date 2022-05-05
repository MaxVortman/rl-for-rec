import pandas as pd
from datasets import (
    FixedLengthDatasetTrain,
    FixedLengthDatasetTest,
    FixedLengthDatasetCollator,
)
from torch.utils.data import DataLoader
import time
import torch
from training.progressbar import tqdm
from training.losses import compute_td_loss
from training.metrics import ndcg
from training.utils import t2d, seed_all, log_metrics
from training.predictions import direct_predict
from models.rl_models import DQN
import torch.optim as optim


VALID_METRICS_TEMPLATE_STR = "NDCG@10 - {:.3f} NDCG@50 - {:.3f} NDCG@100 - {:.3f}"
TRAIN_METRICS_TEMPLATE_STR = "loss - {:.3f}"


def get_sequences(path):
    data = pd.read_csv(path)
    grouped_and_sorted = data.groupby("uid").apply(
        lambda x: list(x.sort_values(by=["date"])["sid"])
    )
    sequences = grouped_and_sorted.values
    return sequences[:10000]


def get_loaders(
    train_path,
    valid_tr_path=None,
    valid_te_path=None,
    test_tr_path=None,
    test_te_path=None,
    batch_size=32,
    window_size=5,
    padding_idx=0,
):
    train_sequences = get_sequences(train_path)
    rewards = [[1 for _ in s] for s in train_sequences]
    train_dataset = FixedLengthDatasetTrain(
        sequences=train_sequences, rewards=rewards, window_size=window_size,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=FixedLengthDatasetCollator(),
    )

    if not valid_tr_path or not valid_te_path:
        return train_loader

    valid_tr_sequences = get_sequences(valid_tr_path)
    valid_te_sequences = get_sequences(valid_te_path)
    valid_dataset = FixedLengthDatasetTest(
        sequences_tr=valid_tr_sequences,
        sequences_te=valid_te_sequences,
        window_size=window_size,
        padding_idx=padding_idx
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=FixedLengthDatasetCollator(),
    )

    if not test_tr_path or not test_te_path:
        return train_loader, valid_loader

    test_tr_sequences = get_sequences(test_tr_path)
    test_te_sequences = get_sequences(test_te_path)
    test_dataset = FixedLengthDatasetTest(
        sequences_tr=test_tr_sequences,
        sequences_te=test_te_sequences,
        window_size=window_size,
        padding_idx=padding_idx
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=FixedLengthDatasetCollator(),
    )

    return train_loader, valid_loader, test_loader


def train_fn(
    model, loader, device, optimizer, gamma=0.9, scheduler=None, accumulation_steps=1
):
    model.train()

    metrics = {
        "loss": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            batch = t2d(batch, device)
            optimizer.zero_grad()
            loss = compute_td_loss(model, batch, gamma)
            metrics["loss"] += loss.detach().item()

            progress.set_postfix_str(
                TRAIN_METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
                )
            )
            progress.update(1)

            loss.backward()
            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

    for k in metrics.keys():
        metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def valid_fn(model, loader, device, items_n):
    model.eval()

    metrics = {
        "NDCG@10": 0.0,
        "NDCG@50": 0.0,
        "NDCG@100": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            state, te = t2d(batch, device)

            metrics["NDCG@10"] += ndcg(
                true=te, pred=direct_predict(model, state, 10), items_n=items_n, k=10
            )
            metrics["NDCG@50"] += ndcg(
                true=te, pred=direct_predict(model, state, 50), items_n=items_n, k=50
            )
            metrics["NDCG@100"] += ndcg(
                true=te, pred=direct_predict(model, state, 100), items_n=items_n, k=100
            )

            progress.set_postfix_str(
                VALID_METRICS_TEMPLATE_STR.format(
                    metrics["NDCG@10"] / (idx + 1),
                    metrics["NDCG@50"] / (idx + 1),
                    metrics["NDCG@100"] / (idx + 1),
                )
            )
            progress.update(1)

    for k in metrics.keys():
        metrics[k] /= n_batches

    return metrics


def experiment(n_epochs, device, action_n, padding_idx, batch_size=256, window_size=5, seed=23):
    print("Experiment has been started")
    seed_all(seed)
    train_loader, valid_loader = get_loaders(
        train_path="prepared_data/train.csv",
        valid_te_path="prepared_data/validation_te.csv",
        valid_tr_path="prepared_data/validation_tr.csv",
        # train_path="prepared_data/pipeline_test_data.csv",
        # valid_te_path="prepared_data/pipeline_test_data.csv",
        # valid_tr_path="prepared_data/pipeline_test_data.csv",
        batch_size=batch_size,
        window_size=window_size,
        padding_idx=padding_idx,
    )
    print("Date is loaded succesfully")
    model = DQN(action_n=action_n, embedding_dim=32, seq_size=window_size, padding_idx=padding_idx)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    print("Training...")

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

        train_metrics = train_fn(
            model,
            train_loader,
            device,
            optimizer,
        )

        log_metrics(train_metrics, "Train")

        valid_metrics = valid_fn(model, valid_loader, device, action_n)

        log_metrics(valid_metrics, "Valid")


if __name__ == "__main__":
    actions_n = 17769
    padding_idx = actions_n
    actions_n += 1
    experiment(1, "cpu", actions_n, padding_idx)
