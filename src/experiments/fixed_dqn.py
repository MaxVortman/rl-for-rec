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
import pickle


METRICS_TEMPLATE_STR = (
    "loss - {:.3f} NDCG@10 - {:.3f} NDCG@50 - {:.3f} NDCG@100 - {:.3f}"
)


def get_loaders(
    seq_dataset_path,
    num_workers=0,
    batch_size=32,
    window_size=5,
    padding_idx=0,
):
    with open(seq_dataset_path, "rb") as f:
        seq_dataset = pickle.load(f)

    train_sequences = seq_dataset["train"]
    train_dataset = FixedLengthDatasetTrain(
        sequences=train_sequences,
        rewards=[[1 for _ in s] for s in train_sequences],
        window_size=window_size,
        padding_idx=padding_idx,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=FixedLengthDatasetCollator(),
        num_workers=num_workers,
    )

    valid_dataset = FixedLengthDatasetTest(
        sequences_tr=seq_dataset["validation_tr"],
        sequences_te=seq_dataset["validation_te"],
        rewards=[1] * len(seq_dataset["validation_tr"]),
        window_size=window_size,
        padding_idx=padding_idx,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=FixedLengthDatasetCollator(),
        num_workers=num_workers,
    )

    test_dataset = FixedLengthDatasetTest(
        sequences_tr=seq_dataset["test_tr"],
        sequences_te=seq_dataset["test_te"],
        rewards=[1] * len(seq_dataset["test_tr"]),
        window_size=window_size,
        padding_idx=padding_idx,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=FixedLengthDatasetCollator(),
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader


def train_fn(
    model,
    loader,
    device,
    optimizer,
    items_n,
    padding_idx,
    gamma=0.9,
    scheduler=None,
    accumulation_steps=1,
):
    model.train()

    metrics = {
        "loss": 0.0,
        "NDCG@10": 0.0,
        "NDCG@50": 0.0,
        "NDCG@100": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            state, action, reward, next_state, done, te = t2d(batch, device)
            optimizer.zero_grad()
            loss = compute_td_loss(
                model, state, action, reward, next_state, done, gamma
            )
            metrics["loss"] += loss.detach().item()

            prediction = direct_predict(model, state)

            metrics["NDCG@10"] += ndcg(
                true=te,
                pred=prediction,
                items_n=items_n,
                k=10,
                padding_idx=padding_idx,
                device=device,
            ).detach().item()
            metrics["NDCG@50"] += ndcg(
                true=te,
                pred=prediction,
                items_n=items_n,
                k=50,
                padding_idx=padding_idx,
                device=device,
            ).detach().item()
            metrics["NDCG@100"] += ndcg(
                true=te,
                pred=prediction,
                items_n=items_n,
                k=100,
                padding_idx=padding_idx,
                device=device,
            ).detach().item()

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
                    metrics["NDCG@10"] / (idx + 1),
                    metrics["NDCG@50"] / (idx + 1),
                    metrics["NDCG@100"] / (idx + 1),
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
def valid_fn(model, loader, device, items_n, padding_idx, gamma=0.9):
    model.eval()

    metrics = {
        "loss": 0.0,
        "NDCG@10": 0.0,
        "NDCG@50": 0.0,
        "NDCG@100": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            state, action, reward, next_state, done, te = t2d(batch, device)

            loss = compute_td_loss(
                model, state, action, reward, next_state, done, gamma
            )
            metrics["loss"] += loss.detach().item()

            prediction = direct_predict(model, state)

            metrics["NDCG@10"] += ndcg(
                true=te,
                pred=prediction,
                items_n=items_n,
                k=10,
                padding_idx=padding_idx,
                device=device,
            ).detach().item()
            metrics["NDCG@50"] += ndcg(
                true=te,
                pred=prediction,
                items_n=items_n,
                k=50,
                padding_idx=padding_idx,
                device=device,
            ).detach().item()
            metrics["NDCG@100"] += ndcg(
                true=te,
                pred=prediction,
                items_n=items_n,
                k=100,
                padding_idx=padding_idx,
                device=device,
            ).detach().item()

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
                    metrics["NDCG@10"] / (idx + 1),
                    metrics["NDCG@50"] / (idx + 1),
                    metrics["NDCG@100"] / (idx + 1),
                )
            )
            progress.update(1)

    for k in metrics.keys():
        metrics[k] /= n_batches

    return metrics


def experiment(
    n_epochs,
    device,
    prepared_data_path,
    num_workers=0,
    embedding_dim=32,
    batch_size=256,
    window_size=5,
    seed=23,
):
    with open(prepared_data_path + "/unique_sid.txt", "r") as f:
        action_n = len(f.readlines())
    padding_idx = action_n
    print(f"Number of possible actions is {action_n}")
    print(f"Padding index is {padding_idx}")

    print("Experiment has been started")
    seed_all(seed)
    train_loader, valid_loader, test_loader = get_loaders(
        seq_dataset_path=f"{prepared_data_path}/seq_dataset.pkl",
        batch_size=batch_size,
        window_size=window_size,
        padding_idx=padding_idx,
        num_workers=num_workers,
    )
    print("Data is loaded succesfully")
    model = DQN(
        action_n=action_n,
        embedding_dim=embedding_dim,
        seq_size=window_size,
        padding_idx=padding_idx,
    )
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
            items_n=action_n,
            padding_idx=padding_idx,
        )

        log_metrics(train_metrics, "Train")

        valid_metrics = valid_fn(
            model, valid_loader, device, action_n, padding_idx=padding_idx
        )

        log_metrics(valid_metrics, "Valid")


if __name__ == "__main__":
    experiment(n_epochs=1, device="cpu", prepared_data_path="prepared_data")
