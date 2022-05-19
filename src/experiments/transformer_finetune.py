from datasets.seq_reward_datasets import (
    SeqRewardDatasetTest,
    SeqRewardDatasetTrain,
    SeqRewardTestDatasetCollator,
    SeqRewardTrainDatasetCollator,
)
from torch.utils.data import DataLoader
import time
import torch
from training.progressbar import tqdm
from training.utils import t2d, seed_all, log_metrics
from training.checkpoint import CheckpointManager, make_checkpoint, load_embedding
from training.predictions import direct_predict_transformer, prepare_true_matrix
from training.metrics import ndcg_rewards
from models.transformer import (
    DqnFreezeTransformer,
    TransformerEmbedding,
    generate_square_subsequent_mask,
    create_pad_mask,
)
from training.losses import compute_td_loss_transformer_finetune
import torch.optim as optim
import pickle


METRICS_TEMPLATE_STR = "loss - {:.3f}"
TEST_METRICS_TEMPLATE_STR = "loss - {:.3f} direct_NDCG@100 - {:.3f}"


def get_loaders(
    seq_reward_path,
    num_workers=0,
    batch_size=32,
    max_size=512,
    padding_idx=0,
):
    with open(seq_reward_path, "rb") as f:
        seq_reward = pickle.load(f)

    collate_fn_train = SeqRewardTrainDatasetCollator(
        max_size=max_size, padding_idx=padding_idx
    )
    collate_fn_test = SeqRewardTestDatasetCollator(
        max_size=max_size, padding_idx=padding_idx
    )

    train_dataset = SeqRewardDatasetTrain(
        sequences=seq_reward["train_data_seq"],
        rewards=seq_reward["train_data_rewards"],
        max_size=max_size,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_train,
        num_workers=num_workers,
    )

    valid_dataset = SeqRewardDatasetTest(
        sequences_tr=seq_reward["vad_data_tr_seq"],
        rewards_tr=seq_reward["vad_data_tr_rewards"],
        sequences_te=seq_reward["vad_data_te_seq"],
        rewards_te=seq_reward["vad_data_te_rewards"],
        max_size=max_size,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_test,
        num_workers=num_workers,
    )

    test_dataset = SeqRewardDatasetTest(
        sequences_tr=seq_reward["test_data_tr_seq"],
        rewards_tr=seq_reward["test_data_tr_rewards"],
        sequences_te=seq_reward["test_data_te_seq"],
        rewards_te=seq_reward["test_data_te_rewards"],
        max_size=max_size,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_test,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader


def train_fn(
    model,
    loader,
    device,
    optimizer,
    max_size,
    padding_idx,
    gamma=0.9,
    scheduler=None,
    accumulation_steps=1,
):
    model.train()

    metrics = {"loss": 0.0}
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            batch = t2d(batch, device)

            optimizer.zero_grad()
            loss = compute_td_loss_transformer_finetune(
                model, batch, gamma, src_mask, padding_idx
            )
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
def valid_fn(
    model,
    loader,
    device,
    max_size,
    padding_idx,
    items_n,
    gamma=0.9,
):
    model.eval()

    metrics = {"loss": 0.0, "direct_NDCG@100": 0.0}
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes, rewards_tes, tr_last_ind = batch
            loss_batch = t2d(loss_batch, device)
            tr_last_ind = tr_last_ind.to(device)

            loss = compute_td_loss_transformer_finetune(
                model, loss_batch, gamma, src_mask, padding_idx
            )
            loss_item = loss.detach().item()
            metrics["loss"] += loss_item

            states = loss_batch[0]
            direct_prediction = direct_predict_transformer(
                model, states, src_mask, padding_idx, tr_last_ind, trs=trs
            )

            true = prepare_true_matrix(tes, rewards_tes, items_n, device)
            direct_ndcg100 = ndcg_rewards(true, direct_prediction, k=100)
            metrics["direct_NDCG@100"] += direct_ndcg100

            progress.set_postfix_str(
                TEST_METRICS_TEMPLATE_STR.format(
                    metrics["loss"] / (idx + 1),
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
    checkpoint_dir,
    num_workers=0,
    batch_size=256,
    seed=23,
    lr=1e-3,
    max_size=512,
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

    print("Experiment has been started")
    seed_all(seed)
    train_loader, valid_loader, test_loader = get_loaders(
        seq_reward_path=f"{prepared_data_path}/seq_rewards.pkl",
        batch_size=batch_size,
        padding_idx=padding_idx,
        num_workers=num_workers,
        max_size=max_size,
    )
    print("Data is loaded succesfully")
    transformer_embedding = load_embedding(
        checkpoint_dir=checkpoint_dir, model_class=TransformerEmbedding
    )
    model = DqnFreezeTransformer(
        transformer_embedding=transformer_embedding,
        ntoken=action_n,
        d_model=transformer_embedding.d_model,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
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
            max_size=max_size,
            padding_idx=padding_idx,
            scheduler=scheduler,
        )

        log_metrics(train_metrics, "Train")

        valid_metrics = valid_fn(
            model,
            valid_loader,
            device,
            max_size=max_size,
            padding_idx=padding_idx,
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
                metrics={"train": train_metrics, "valid": valid_metrics},
                epoch_start_time=epoch_start_time,
            ),
        )


if __name__ == "__main__":
    experiment(
        n_epochs=1,
        device="cpu",
        prepared_data_path="prepared_whole_data",
        logdir="logs/transformer_finetune",
        checkpoint_dir="logs/transformer_embedding",
        batch_size=2,
        max_size=512,
    )
