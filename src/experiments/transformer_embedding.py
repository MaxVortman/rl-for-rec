from datasets.transformer_datasets import (
    TransformerDatasetTrain,
    TransformerDatasetTrainCollator,
)
from torch.utils.data import DataLoader
import time
import torch
from training.progressbar import tqdm
from training.utils import t2d, seed_all, log_metrics
from training.checkpoint import CheckpointManager, make_checkpoint
from models.transformer import (
    TransformerModel,
    generate_square_subsequent_mask,
    create_pad_mask,
)
import torch.optim as optim
import pickle


METRICS_TEMPLATE_STR = "loss - {:.3f}"


def get_loaders(
    prepared_data_path,
    num_workers=0,
    batch_size=32,
    max_size=512,
    padding_idx=0,
):
    with open(f"{prepared_data_path}/train_data_seq.pkl", "rb") as f:
        train_data_seq = pickle.load(f)

    with open(f"{prepared_data_path}/vad_data_tr_seq.pkl", "rb") as f:
        vad_data_tr_seq = pickle.load(f)

    collate_fn = TransformerDatasetTrainCollator(
        max_size=max_size, padding_idx=padding_idx
    )

    train_dataset = TransformerDatasetTrain(
        sequences=train_data_seq,
        max_size=max_size,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    valid_dataset = TransformerDatasetTrain(
        sequences=vad_data_tr_seq,
        max_size=max_size,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


def train_fn(
    model,
    loader,
    device,
    optimizer,
    loss_fn,
    max_size,
    padding_idx,
    scheduler=None,
    accumulation_steps=1,
):
    model.train()

    metrics = {"loss": 0.0}
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            sources, targets = t2d(batch, device)
            pad_mask = create_pad_mask(matrix=sources, pad_token=padding_idx)
            output = model(
                src=sources, src_mask=src_mask, src_key_padding_mask=pad_mask
            )

            optimizer.zero_grad()
            loss = loss_fn(output, targets)
            loss_item = loss.detach().item()
            metrics["loss"] += loss_item

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
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
def valid_fn(model, loader, device, loss_fn, max_size, padding_idx):
    model.eval()

    metrics = {"loss": 0.0}
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            sources, targets = t2d(batch, device)
            pad_mask = create_pad_mask(matrix=sources, pad_token=padding_idx)
            output = model(
                src=sources, src_mask=src_mask, src_key_padding_mask=pad_mask
            )

            loss = loss_fn(output, targets)
            loss_item = loss.detach().item()
            metrics["loss"] += loss_item

            progress.set_postfix_str(
                METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
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
    batch_size=256,
    seed=23,
    lr=1e-3,
    max_size=512,
    d_model=512,
    d_hid=512,
    n_head=8,
    num_encoder_layers=6,
    dropout_rate=0.5,
):
    with open(prepared_data_path + "/unique_sid.txt", "r") as f:
        action_n = len(f.readlines())
    padding_idx = 0
    print(f"Number of possible actions is {action_n}")
    print(f"Padding index is {padding_idx}")

    main_metric = "loss"

    checkpointer = CheckpointManager(
        logdir=logdir,
        metric=main_metric,
        metric_minimization=True,
        save_n_best=1,
    )

    model_config = {
        "name": "TransformerEmbedding",
        "args": dict(
            ntoken=action_n,
            d_model=d_model,
            padding_idx=padding_idx,
            nhead=n_head,
            nlayers=num_encoder_layers,
            dropout=dropout_rate,
            d_hid=d_hid,
        ),
    }

    print("Experiment has been started")
    seed_all(seed)
    train_loader, valid_loader = get_loaders(
        prepared_data_path=prepared_data_path,
        batch_size=batch_size,
        padding_idx=padding_idx,
        num_workers=num_workers,
        max_size=max_size,
    )
    print("Data is loaded succesfully")
    model = TransformerModel(**model_config["args"])
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = None
    print("Training...")

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

        train_metrics = train_fn(
            model,
            train_loader,
            device,
            optimizer,
            loss_fn=loss_fn,
            max_size=max_size,
            padding_idx=padding_idx,
            scheduler=scheduler,
        )

        log_metrics(train_metrics, "Train")

        valid_metrics = valid_fn(
            model,
            valid_loader,
            device,
            loss_fn=loss_fn,
            max_size=max_size,
            padding_idx=padding_idx,
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
        logdir="logs/transformer_embedding",
        batch_size=512,
        max_size=8,
    )
