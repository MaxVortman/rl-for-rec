import numpy as np
from scipy.sparse import csr_matrix
import torch


def ndcg(true, pred, items_n, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in true indicate 0 relevance
    """
    pred = pred.detach().numpy()
    batch_size = pred.shape[0]
    true_full = torch.zeros(size=(batch_size, items_n))
    true_full = (
        true_full.scatter_(1, true, torch.ones_like(true, dtype=true_full.dtype))
        .detach()
        .numpy()
    )
    tp = 1.0 / np.log2(np.arange(2, k + 2))
    DCG = (np.take_along_axis(true_full, pred, axis=1) * tp).sum(axis=1)
    IDCG = np.array(
        [(tp[: min(int(n), k)]).sum() for n in (true_full != 0).sum(axis=1)]
    )
    return (DCG / IDCG).mean()
