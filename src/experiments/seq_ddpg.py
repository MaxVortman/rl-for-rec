from datasets.seq_datasets import (
    SeqDatasetTrain,
    SeqDatasetTest,
    SeqDatasetCollator,
)
from torch.utils.data import DataLoader
import time
import torch
import torch.nn as nn
from training.progressbar import tqdm
from training.losses import ddpg_loss
from training.metrics import ndcg
from training.utils import t2d, seed_all, log_metrics, soft_update
from training.predictions import direct_predict, prepare_true_matrix
from models.a2c import Actor, Critic
import torch.optim as optim
import pickle


METRICS_TEMPLATE_STR = "policy loss - {:.3f} value loss - {:.3f} NDCG@10 - {:.3f} NDCG@50 - {:.3f} NDCG@100 - {:.3f}"


def get_loaders(
    seq_dataset_path,
    num_workers=0,
    batch_size=32,
    max_tr_size=512,
    padding_idx=0,
):
    with open(seq_dataset_path, "rb") as f:
        seq_dataset = pickle.load(f)

    collate_fn = SeqDatasetCollator()

    train_sequences = seq_dataset["train"]
    train_dataset = SeqDatasetTrain(
        sequences=train_sequences,
        max_tr_size=max_tr_size,
        padding_idx=padding_idx,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    valid_dataset = SeqDatasetTest(
        sequences_tr=seq_dataset["validation_tr"],
        sequences_te=seq_dataset["validation_te"],
        max_tr_size=max_tr_size,
        padding_idx=padding_idx,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    test_dataset = SeqDatasetTest(
        sequences_tr=seq_dataset["test_tr"],
        sequences_te=seq_dataset["test_te"],
        max_tr_size=max_tr_size,
        padding_idx=padding_idx,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader


def train_fn(
    value_net,
    policy_net,
    target_value_net,
    target_policy_net,
    value_criterion,
    policy_optimizer,
    value_optimizer,
    min_value,
    max_value,
    policy_step,
    loader,
    device,
    items_n,
    gamma=0.9,
    soft_tau=1e-2,
    count_metrics_steps=1,
):
    value_net.train()
    policy_net.train()

    metrics = {
        "policy loss": 0.0,
        "value loss": 0.0,
        "NDCG@10": 0.0,
        "NDCG@50": 0.0,
        "NDCG@100": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes = batch
            state, action, reward, next_state, done = t2d(loss_batch, device)

            policy_loss, value_loss = ddpg_loss(
                state,
                action,
                reward,
                next_state,
                done,
                value_net,
                policy_net,
                target_value_net,
                target_policy_net,
                value_criterion,
                gamma=gamma,
                min_value=min_value,
                max_value=max_value,
            )

            metrics["policy loss"] += policy_loss.detach().item()
            metrics["value loss"] += value_loss.detach().item()

            if (idx + 1) % count_metrics_steps == 0:
                prediction = direct_predict(target_policy_net, state, trs)
                true = prepare_true_matrix(tes, items_n, device)
                ndcg10, ndcg50, ndcg100 = (
                    ndcg(true, prediction, 10),
                    ndcg(true, prediction, 50),
                    ndcg(true, prediction, 100),
                )
                metrics["NDCG@10"] = ndcg10
                metrics["NDCG@50"] = ndcg50
                metrics["NDCG@100"] = ndcg100

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
                    metrics["policy loss"] / (idx + 1),
                    metrics["value loss"] / (idx + 1),
                    metrics["NDCG@10"],
                    metrics["NDCG@50"],
                    metrics["NDCG@100"],
                )
            )
            progress.update(1)

            if (idx + 1) % policy_step == 0:
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), -1, 1)
                policy_optimizer.step()

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                soft_update(target_value_net, value_net, soft_tau)
                soft_update(target_policy_net, policy_net, soft_tau)
            else:
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

    for k in metrics.keys():
        metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def valid_fn(
    value_net,
    policy_net,
    target_value_net,
    target_policy_net,
    value_criterion,
    min_value,
    max_value,
    loader,
    device,
    items_n,
    gamma=0.9,
):
    value_net.eval()
    policy_net.eval()

    metrics = {
        "policy loss": 0.0,
        "value loss": 0.0,
        "NDCG@10": 0.0,
        "NDCG@50": 0.0,
        "NDCG@100": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes = batch
            state, action, reward, next_state, done = t2d(loss_batch, device)

            policy_loss, value_loss = ddpg_loss(
                state,
                action,
                reward,
                next_state,
                done,
                value_net,
                policy_net,
                target_value_net,
                target_policy_net,
                value_criterion,
                gamma=gamma,
                min_value=min_value,
                max_value=max_value,
            )

            metrics["policy loss"] += policy_loss.detach().item()
            metrics["value loss"] += value_loss.detach().item()
            prediction = direct_predict(target_policy_net, state, trs)
            true = prepare_true_matrix(tes, items_n, device)
            ndcg10, ndcg50, ndcg100 = (
                ndcg(true, prediction, 10),
                ndcg(true, prediction, 50),
                ndcg(true, prediction, 100),
            )
            metrics["NDCG@10"] += ndcg10
            metrics["NDCG@50"] += ndcg50
            metrics["NDCG@100"] += ndcg100

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
                    metrics["policy loss"] / (idx + 1),
                    metrics["value loss"] / (idx + 1),
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
    seed=23,
    count_metrics_steps=1,
    value_lr=1e-3,
    policy_lr=1e-4,
    max_tr_size=512,
    hidden_lstm_size=64,
    hidden_gru_size=64,
    dropout_rate=0.1,
    policy_step=10,
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
        padding_idx=padding_idx,
        num_workers=num_workers,
        max_tr_size=max_tr_size,
    )
    print("Data is loaded succesfully")
    value_net = Critic(
        action_n=action_n,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        hidden_lstm_size=hidden_lstm_size,
        hidden_gru_size=hidden_gru_size,
        dropout_rate=dropout_rate,
    )
    policy_net = Actor(
        action_n=action_n,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        hidden_lstm_size=hidden_lstm_size,
        hidden_gru_size=hidden_gru_size,
        dropout_rate=dropout_rate,
    )
    value_net.to(device)
    policy_net.to(device)

    target_value_net = Critic(
        action_n=action_n,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        hidden_lstm_size=hidden_lstm_size,
        hidden_gru_size=hidden_gru_size,
        dropout_rate=dropout_rate,
    )
    target_policy_net = Actor(
        action_n=action_n,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        hidden_lstm_size=hidden_lstm_size,
        hidden_gru_size=hidden_gru_size,
        dropout_rate=dropout_rate,
    )
    target_value_net.to(device)
    target_policy_net.to(device)

    soft_update(target_value_net, value_net, 1.0)
    soft_update(target_policy_net, policy_net, 1.0)

    target_policy_net.eval()
    target_value_net.eval()

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    value_criterion = nn.MSELoss()
    print("Training...")

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

        train_metrics = train_fn(
            value_net=value_net,
            policy_net=policy_net,
            target_value_net=target_value_net,
            target_policy_net=target_policy_net,
            value_criterion=value_criterion,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            min_value=-10,
            max_value=10,
            policy_step=policy_step,
            loader=train_loader,
            device=device,
            count_metrics_steps=count_metrics_steps,
            items_n=action_n,
        )

        log_metrics(train_metrics, "Train")

        valid_metrics = valid_fn(
            value_net=value_net,
            policy_net=policy_net,
            target_value_net=target_value_net,
            target_policy_net=target_policy_net,
            value_criterion=value_criterion,
            min_value=-10,
            max_value=10,
            loader=valid_loader,
            device=device,
            items_n=action_n,
        )

        log_metrics(valid_metrics, "Valid")


if __name__ == "__main__":
    experiment(
        n_epochs=1, device="cpu", prepared_data_path="prepared_data", batch_size=256
    )
