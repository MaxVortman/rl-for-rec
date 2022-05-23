from datasets.seq_reward_datasets import (
    SeqRewardDatasetTest,
    SeqRewardTestDatasetCollator,
)
from torch.utils.data import DataLoader
import torch
from training.progressbar import tqdm
from training.utils import t2d, log_metrics
from training.checkpoint import load_checkpoint, load_model_config
from training.predictions import direct_predict_transformer, prepare_true_matrix_rewards
from training.metrics import ndcg_rewards
from models.transformer import (
    TransformerModel,
    TransformerEmbeddingFreeze,
    DqnFreezeTransformer,
    generate_square_subsequent_mask,
)
import pickle
import os
import pandas as pd


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

    collate_fn_test = SeqRewardTestDatasetCollator(
        max_size=max_size, padding_idx=padding_idx
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

    return test_loader


@torch.no_grad()
def test_fn(
    model,
    loader,
    device,
    max_size,
    padding_idx,
    items_n,
):
    model.eval()

    metrics = {"direct_NDCG@100": 0.0}
    n_batches = len(loader)

    src_mask = generate_square_subsequent_mask(max_size).to(device)

    with tqdm(total=n_batches, desc="test") as progress:
        for idx, batch in enumerate(loader):
            loss_batch, trs, tes, rewards_tes, tr_last_ind = batch
            loss_batch = t2d(loss_batch, device)
            tr_last_ind = tr_last_ind.to(device)

            states = loss_batch[0]
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


def transformer_test(loader, device, transformer_checkpoint_dir, max_size, padding_idx, items_n):
    model = load_model_config(transformer_checkpoint_dir, TransformerModel)
    best_path = os.path.join(transformer_checkpoint_dir, "best.pth")
    load_checkpoint(best_path, model)
    model.to(device)
    test_metrics = test_fn(
            model,
            loader,
            device,
            max_size=max_size,
            padding_idx=padding_idx,
            items_n=items_n,
        )
    log_metrics(test_metrics, "Test")
    return test_metrics


def transformer_finetuned_test(loader, device, embedding_checkpoint_dir, finetuned_checkpoint_dir, max_size, padding_idx, items_n):
    transformer_embedding = load_model_config(embedding_checkpoint_dir, TransformerEmbeddingFreeze)
    model = DqnFreezeTransformer(
        transformer_embedding=transformer_embedding,
        ntoken=items_n,
        d_model=transformer_embedding.d_model,
    )
    best_path = os.path.join(finetuned_checkpoint_dir, "best.pth")
    load_checkpoint(best_path, model)
    model.to(device)
    test_metrics = test_fn(
            model,
            loader,
            device,
            max_size=max_size,
            padding_idx=padding_idx,
            items_n=items_n,
        )
    log_metrics(test_metrics, "Test")
    return test_metrics


def experiment(
    device,
    prepared_data_path,
    logdir,
    finetuned_checkpoint_dirs=None,
    transformer_checkpoint_dirs=None,
    num_workers=0,
    batch_size=256,
    max_size=512,
):
    with open(prepared_data_path + "/unique_sid.txt", "r") as f:
        action_n = len(f.readlines())
    padding_idx = 0
    print(f"Number of possible actions is {action_n}")
    print(f"Padding index is {padding_idx}")
    test_loader = get_loaders(
        seq_reward_path=f"{prepared_data_path}/seq_rewards.pkl",
        batch_size=batch_size,
        padding_idx=padding_idx,
        num_workers=num_workers,
        max_size=max_size,
    )
    print("Data is loaded succesfully")
    
    print("Testing...")

    metrics = []
    if transformer_checkpoint_dirs:
        for dir in transformer_checkpoint_dirs:
            test_metrics = transformer_test(test_loader, device, dir, max_size, padding_idx, action_n)
            test_metrics["model"] = dir
            metrics.append(test_metrics)

    if finetuned_checkpoint_dirs:
        for emb_dir, trans_dir in finetuned_checkpoint_dirs:
            test_metrics = transformer_finetuned_test(test_loader, device, emb_dir, trans_dir, max_size, padding_idx, action_n)
            test_metrics["model"] = trans_dir
            metrics.append(test_metrics)

    if metrics:
        metrics_df = pd.DataFrame.from_records(metrics)
        print(metrics_df)
        metrics_df.to_csv(logdir + "/test_metrics.csv")

    


if __name__ == "__main__":
    experiment(
        device="cpu",
        prepared_data_path="prepared_whole_data",
        logdir="logs",
        transformer_checkpoint_dirs=["logs/transformer_embedding"],
        batch_size=2,
        max_size=512,
    )
