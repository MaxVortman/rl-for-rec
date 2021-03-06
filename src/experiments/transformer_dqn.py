from datasets.seq_datasets import (
    SeqDatasetTrain,
    SeqDatasetTest,
    SeqDatasetCollator,
)
from torch.utils.data import DataLoader
import time
import torch
from training.progressbar import tqdm
from training.losses import compute_td_loss_transformer
from training.metrics import ndcg, ndcg_chain
from training.utils import t2d, seed_all, log_metrics
from training.predictions import (
    prepare_true_matrix,
    direct_predict_transformer,
    chain_predict_transformer,
)
from models.transformer import TransformerModel, generate_square_subsequent_mask
import torch.optim as optim
import pickle


METRICS_TEMPLATE_STR = "loss - {:.3f} direct_NDCG@100 - {:.3f} chain_NDCG@100 - {:.3f}"


def get_loaders(
    seq_dataset_path,
    num_workers=0,
    batch_size=32,
    max_tr_size=512,
    padding_idx=0,
):
    with open(seq_dataset_path, "rb") as f:
        seq_dataset = pickle.load(f)

    collate_fn = SeqDatasetCollator(max_size=max_tr_size, padding_idx=padding_idx)

    train_sequences = seq_dataset["train"]
    train_dataset = SeqDatasetTrain(
        sequences=train_sequences,
        max_tr_size=max_tr_size,
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
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader


def train_fn(
    model,
    loader,
    device,
    optimizer,
    items_n,
    max_size,
    padding_idx,
    gamma=0.9,
    scheduler=None,
    accumulation_steps=1,
    count_metrics_steps=1,
):
    model.train()

    metrics = {
        "loss": 0.0,
        "direct_NDCG@100": 0.0,
        "chain_NDCG@100": 0.0,
    }
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes = batch
            state, action, reward, next_state, done = t2d(loss_batch, device)
            optimizer.zero_grad()
            loss = compute_td_loss_transformer(
                model,
                state,
                action,
                reward,
                next_state,
                done,
                gamma,
                src_mask=src_mask,
                padding_idx=padding_idx,
            )
            metrics["loss"] += loss.detach().item()

            if (idx + 1) % count_metrics_steps == 0:
                direct_prediction = direct_predict_transformer(
                    model, state, src_mask, padding_idx, trs=trs
                )
                chain_prediction = chain_predict_transformer(
                    model, state, src_mask, padding_idx, k=100, trs=trs
                )

                true = prepare_true_matrix(tes, items_n, device)
                direct_ndcg100 = ndcg(true, direct_prediction, k=100)
                chain_ndcg100 = ndcg_chain(true, chain_prediction, k=100)
                metrics["direct_NDCG@100"] = direct_ndcg100
                metrics["chain_NDCG@100"] = chain_ndcg100

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
                    metrics["direct_NDCG@100"],
                    metrics["chain_NDCG@100"],
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
def valid_fn(model, loader, device, items_n, max_size, padding_idx, gamma=0.9):
    model.eval()

    metrics = {
        "loss": 0.0,
        "direct_NDCG@100": 0.0,
        "chain_NDCG@100": 0.0,
    }
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes = batch
            state, action, reward, next_state, done = t2d(loss_batch, device)

            loss = compute_td_loss_transformer(
                model,
                state,
                action,
                reward,
                next_state,
                done,
                gamma,
                src_mask=src_mask,
                padding_idx=padding_idx,
            )
            metrics["loss"] += loss.detach().item()

            direct_prediction = direct_predict_transformer(
                model, state, src_mask, padding_idx, trs=trs
            )
            chain_prediction = chain_predict_transformer(
                model, state, src_mask, padding_idx, k=100, trs=trs
            )

            true = prepare_true_matrix(tes, items_n, device)
            direct_ndcg100 = ndcg(true, direct_prediction, k=100)
            chain_ndcg100 = ndcg_chain(true, chain_prediction, k=100)
            metrics["direct_NDCG@100"] += direct_ndcg100
            metrics["chain_NDCG@100"] += chain_ndcg100

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
                    metrics["direct_NDCG@100"] / (idx + 1),
                    metrics["chain_NDCG@100"] / (idx + 1),
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
    batch_size=256,
    seed=23,
    count_metrics_steps=1,
    lr=1e-3,
    max_tr_size=512,
    d_model=512,
    d_hid=512,
    n_head=8,
    num_encoder_layers=6,
    dropout_rate=0.5,
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
    model = TransformerModel(
        ntoken=action_n,
        d_model=d_model,
        padding_idx=padding_idx,
        nhead=n_head,
        nlayers=num_encoder_layers,
        dropout=dropout_rate,
        d_hid=d_hid,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    print("Training...")

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

        train_metrics = train_fn(
            model,
            train_loader,
            device,
            optimizer,
            count_metrics_steps=count_metrics_steps,
            items_n=action_n,
            max_size=max_tr_size,
            padding_idx=padding_idx,
        )

        log_metrics(train_metrics, "Train")

        valid_metrics = valid_fn(
            model,
            valid_loader,
            device,
            items_n=action_n,
            max_size=max_tr_size,
            padding_idx=padding_idx,
        )

        log_metrics(valid_metrics, "Valid")


if __name__ == "__main__":
    experiment(
        n_epochs=1,
        device="cpu",
        prepared_data_path="prepared_data",
        batch_size=2,
        max_tr_size=8,
    )
