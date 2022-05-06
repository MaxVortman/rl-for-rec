import numpy as np
from scipy.sparse import csr_matrix
import torch


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
