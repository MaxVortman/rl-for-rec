import torch


def direct_predict(model, state, trs=None):
    output = model(state)
    if trs:
        for i, tr in enumerate(trs):
            output[i, tr] = 0
    else:
        output = output.scatter_(1, state, torch.zeros_like(state, dtype=output.dtype))

    return output


def chain_predict(model, state, k, trs=None):
    prediction = torch.zeros(size=(state.size(0), k), device=state.device, dtype=torch.long)
    for i in torch.arange(k, device=state.device):
        output = direct_predict(model, state, trs)
        action = torch.argmax(output, dim=1, keepdim=True)
        prediction[:, i] = action
        state = torch.cat([state[:, 1:], action], dim=1)

    return prediction


def prepare_true_matrix(tes, items_n, device):
    true_matrix = torch.zeros(
        size=(len(tes), items_n + 1), device=device
    )  # + padding index

    for i, te in enumerate(tes):
        true_matrix[i, te] = 1

    return true_matrix
