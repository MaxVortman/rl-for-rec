import torch


def direct_predict(model, state, k):
    output = model(state)
    output = output.scatter_(1, state, torch.zeros_like(state, dtype=output.dtype))
    topk_inds = torch.topk(output, k, dim=1).indices
    return topk_inds
