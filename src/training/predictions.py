import torch


def direct_predict(model, state):
    output = model(state)
    output = output.scatter_(1, state, torch.zeros_like(state, dtype=output.dtype))

    return output


def prepare_true_matrix(tes, items_n, device):
    true_matrix = torch.zeros(
        size=(len(tes), items_n + 1) # + padding index
    )

    for i, te in enumerate(tes):
        true_matrix[i, te] = 1

    true_matrix.to(device)

    return true_matrix
