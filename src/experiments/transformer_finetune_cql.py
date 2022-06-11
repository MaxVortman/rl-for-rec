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
from training.utils import t2d, seed_all, log_metrics, soft_update
from training.checkpoint import CheckpointManager, make_checkpoint, load_embedding
from training.predictions import direct_predict_transformer, prepare_true_matrix_rewards
from training.metrics import ndcg_rewards
from models.transformer import (
    DqnTransformerEmbedding,
    TransformerEmbeddingFreeze,
    DqnFreezeTransformer,
    TransformerEmbedding,
    generate_square_subsequent_mask,
    create_pad_mask,
)
from training.losses import compute_cql_loss_transformer
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import pickle


TRAIN_METRICS_TEMPLATE_STR = "loss - {:.3f} td_loss - {:.3f} cql_loss - {:.3f}"
TEST_METRICS_TEMPLATE_STR = "direct_NDCG@100 - {:.3f}"


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

    return train_loader, valid_loader


def train_fn(
    model,
    target_model,
    loader,
    device,
    optimizer,
    max_size,
    padding_idx,
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

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="train") as progress:
        for idx, batch in enumerate(loader):
            batch = t2d(batch, device)

            optimizer.zero_grad()
            loss, cql_loss, td_loss = compute_cql_loss_transformer(
                model, target_model, batch, gamma, src_mask, padding_idx
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
def valid_fn(
    model,
    loader,
    device,
    max_size,
    padding_idx,
    items_n,
):
    model.eval()

    metrics = {
        "direct_NDCG@100": 0.0,
    }
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="valid") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes, rewards_tes, tr_last_ind = batch
            tr_last_ind = tr_last_ind.to(device)
            states = loss_batch[0].to(device)
            direct_prediction = direct_predict_transformer(
                model, states, src_mask, padding_idx, tr_last_ind, trs=trs
            )

            true = prepare_true_matrix_rewards(tes, rewards_tes, items_n, device)
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
    checkpoint_dir,
    num_workers=0,
    batch_size=256,
    seed=23,
    lr=1e-3,
    max_size=512,
    soft_tau: float = 1e-3,
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
    train_loader, valid_loader = get_loaders(
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
    model_config = {
        "name": "TransformerFinetuneCQL",
        "args": dict(
            transformer_embedding=transformer_embedding,
        ntoken=action_n,
        d_model=transformer_embedding.d_model,
        ),
    }
    model = DqnFreezeTransformer(**model_config["args"])
    target_model = DqnFreezeTransformer(**model_config["args"])
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
            max_size=max_size,
            padding_idx=padding_idx,
            scheduler=scheduler,
            soft_tau=soft_tau,
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
        logdir="logs/transformer_finetune",
        checkpoint_dir="logs/transformer_embedding",
        batch_size=2,
        max_size=512,
    )
