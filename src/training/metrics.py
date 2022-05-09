import torch

from ranking_metrics_torch.cumulative_gain import ndcg_at


def ndcg(true, pred, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in true indicate 0 relevance
    """
    print(pred.device)
    pred_topk = torch.topk(pred, k, dim=1).indices
    tp = 1.0 / torch.log2(torch.arange(2, k + 2, device=pred.device))
    print(true.device)
    print(pred_topk.device)
    DCG = (torch.take_along_dim(true, pred_topk, dim=1) * tp).sum(dim=1)
    IDCG = torch.tensor(
        [(tp[: min(int(n), k)]).sum() for n in (true != 0).sum(dim=1)],
        device=pred.device,
    )
    return (DCG / IDCG).mean(0).detach().item()


def ndcg_lib(ks, true, pred):
    ndcgs = (
        ndcg_at(
            ks=torch.tensor(ks, device=pred.device, dtype=torch.int),
            scores=pred,
            labels=true,
        )
        .mean(0)
        .detach()
    )
    return [i.item() for i in ndcgs]
