import torch

from ranking_metrics_torch.cumulative_gain import ndcg_at


def ndcg(true, pred, items_n, padding_idx, device, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in true indicate 0 relevance
    """
    pred = torch.topk(pred, k, dim=1).indices
    batch_size = pred.size()[0]
    true_full = torch.zeros(
        size=(batch_size, items_n + 1), device=device
    )  # + padding index
    true_full = true_full.scatter_(
        1, true, torch.ones_like(true, dtype=true_full.dtype, device=device)
    )
    true_full[:, padding_idx] = 0  # set padding index to 0
    tp = 1.0 / torch.log2(torch.arange(2, k + 2, device=device))
    DCG = (torch.take_along_dim(true_full, pred, dim=1) * tp).sum(dim=1)
    IDCG = torch.tensor(
        [(tp[: min(int(n), k)]).sum() for n in (true_full != 0).sum(dim=1)],
        device=device,
    )
    return (DCG / IDCG).mean()


def ndcg_lib(ks, true, pred, items_n, padding_idx):
    batch_size = pred.size()[0]
    true_full = torch.zeros(
        size=(batch_size, items_n + 1), device=pred.device
    )  # + padding index
    true_full = true_full.scatter_(
        1, true, torch.ones_like(true, dtype=true_full.dtype, device=pred.device)
    )
    true_full[:, padding_idx] = 0  # set padding index to 0

    ndcgs = (
        ndcg_at(
            ks=torch.tensor(ks, device=pred.device, dtype=torch.int),
            scores=pred,
            labels=true_full,
        )
        .mean(0)
        .detach()
    )
    return [i.item() for i in ndcgs]
