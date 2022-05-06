import torch


def direct_predict(model, state):
    output = model(state)
    output = output.scatter_(1, state, torch.zeros_like(state, dtype=output.dtype))

    return output
