from datasets.seq_datasets import (
    SeqDatasetTrain,
    SeqDatasetTest,
    SeqDatasetTrainCollator,
    SeqDatasetTestCollator,
)
from torch.utils.data import DataLoader
import time
import torch
from torch.nn.utils import clip_grad_norm_
from training.progressbar import tqdm
from training.losses import compute_cql_loss
from training.metrics import ndcg_rewards
from training.utils import t2d, seed_all, log_metrics, soft_update
from training.predictions import direct_predict, prepare_true_matrix_rewards
from training.checkpoint import CheckpointManager, make_checkpoint
from models.seq_dqns import SeqDQN
import torch.optim as optim
import pickle


TRAIN_METRICS_TEMPLATE_STR = "loss - {:.3f} td_loss - {:.3f} cql_loss - {:.3f}"
TEST_METRICS_TEMPLATE_STR = "direct_NDCG@100 - {:.3f}"


def get_loaders(
    seq_dataset_path,
    num_workers=0,
    batch_size=32,
    max_tr_size=512,
    padding_idx=0,
):
    with open(seq_dataset_path, "rb") as f:
        seq_dataset = pickle.load(f)

    train_dataset = SeqDatasetTrain(
        sequences=seq_dataset["train_data_seq"],
        rewards=seq_dataset["train_data_rewards"],
        max_tr_size=max_tr_size,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=SeqDatasetTrainCollator(
            max_size=max_tr_size, padding_idx=padding_idx
        ),
        num_workers=num_workers,
    )

    valid_dataset = SeqDatasetTest(
        sequences_tr=seq_dataset["vad_data_tr_seq"],
        sequences_te=seq_dataset["vad_data_te_seq"],
        rewards_te=seq_dataset["vad_data_te_rewards"],
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=SeqDatasetTestCollator(
            max_size=max_tr_size, padding_idx=padding_idx
        ),
        num_workers=num_workers,
    )

    return train_loader, valid_loader


def train_fn(
    model,
    target_model,
    loader,
    device,
    optimizer,
    gamma=0.9,
    scheduler=None,
    accumulation_steps=1,
    soft_tau=1e-3,
):
    model.train()

    metrics = {
        "loss": 0.0,
        "td_loss": 0.0,
        "cql_loss": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            state, action, reward, next_state, done = t2d(batch, device)
            optimizer.zero_grad()
            loss, cql_loss, td_loss = compute_cql_loss(
                model, target_model, state, action, reward, next_state, done, gamma
            )
            metrics["loss"] += loss.detach().item()
            metrics["td_loss"] += td_loss.detach().item()
            metrics["cql_loss"] += cql_loss.detach().item()

            progress.set_postfix_str(
                TRAIN_METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
                    metrics["td_loss"] / (idx + 1),
                    metrics["cql_loss"] / (idx + 1),
                )
            )
            progress.update(1)

            loss.backward()

            if (idx + 1) % accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            soft_update(target_model, model, soft_tau)

    for k in metrics.keys():
        metrics[k] /= n_batches

    return metrics


@torch.no_grad()
def valid_fn(model, loader, device, items_n):
    model.eval()

    metrics = {
        "direct_NDCG@100": 0.0,
    }
    n_batches = len(loader)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes, reward_tes = batch
            state = loss_batch[0]
            state = state.to(device)

            direct_prediction = direct_predict(model, state, trs=trs)
            true = prepare_true_matrix_rewards(tes, reward_tes, items_n, device)
            direct_ndcg100 = ndcg_rewards(true, direct_prediction, k=100)
            metrics["direct_NDCG@100"] += direct_ndcg100

            progress.set_postfix_str(
                TEST_METRICS_TEMPLATE_STR.format(
                    metrics["direct_NDCG@100"] / (idx + 1),
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
    logdir,
    num_workers=0,
    embedding_dim=32,
    batch_size=256,
    seed=23,
    lr=1e-3,
    max_tr_size=512,
    hidden_lstm_size: int = 64,
    hidden_gru_size: int = 64,
    dropout_rate: float = 0.1,
):
    with open(prepared_data_path + "/unique_sid.txt", "r") as f:
        action_n = len(f.readlines())
    padding_idx = 0
    print(f"Number of possible actions is {action_n}")
    print(f"Padding index is {padding_idx}")

    main_metric = "direct_NDCG@100"

    checkpointer = CheckpointManager(
        logdir=logdir,
        metric=main_metric,
        metric_minimization=False,
        save_n_best=1,
    )

    model_config = {
        "name": "SeqDQN",
        "args": dict(
            action_n=action_n,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            hidden_lstm_size=hidden_lstm_size,
            hidden_gru_size=hidden_gru_size,
            dropout_rate=dropout_rate,
        ),
    }

    print("Experiment has been started")
    seed_all(seed)
    train_loader, valid_loader = get_loaders(
        seq_dataset_path=f"{prepared_data_path}/seq_rewards.pkl",
        batch_size=batch_size,
        padding_idx=padding_idx,
        num_workers=num_workers,
        max_tr_size=max_tr_size,
    )
    print("Data is loaded succesfully")

    model = SeqDQN(**model_config["args"])
    target_model = SeqDQN(**model_config["args"])
    model.to(device)
    target_model.to(device)

    soft_update(target_model, model, 1.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = None
    print("Training...")

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

        train_metrics = train_fn(
            model,
            target_model,
            train_loader,
            device,
            optimizer,
            scheduler=scheduler,
        )

        log_metrics(train_metrics, "Train")

        valid_metrics = valid_fn(
            model,
            valid_loader,
            device,
            items_n=action_n,
        )

        log_metrics(valid_metrics, "Valid")

        checkpointer.process(
            score=valid_metrics[main_metric],
            epoch=epoch,
            checkpoint=make_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                model_configuration=model_config,
                metrics={"train": train_metrics, "valid": valid_metrics},
                epoch_start_time=epoch_start_time,
            ),
        )


if __name__ == "__main__":
    experiment(
        n_epochs=1,
        device="cpu",
        prepared_data_path="prepared_whole_data",
        batch_size=32,
        max_tr_size=16,
        logdir="logs/seq_dqn",
    )
