import torch
from torch.distributions.categorical import Categorical
from models.transformer import create_pad_mask


def direct_predict(model, state, trs=None):
    output = model(state)
    if trs:
        for i, tr in enumerate(trs):
            output[i, tr] = 0
    else:
        output = output.scatter_(1, state, torch.zeros_like(state, dtype=output.dtype))

    return output


def chain_predict(model, state, k, trs=None):
    actions = list()
    for _ in range(k):
        output = direct_predict(model, state, trs)
        action = torch.argmax(output, dim=1, keepdim=True)
        actions.append(action)
        state = torch.cat([state[:, 1:], action], dim=1)
    prediction = torch.cat(actions, dim=1)
    return prediction


def direct_dist_predict(model, state, trs=None):
    probs = model(state)
    if trs:
        for i, tr in enumerate(trs):
            probs[i, tr] = 0
    else:
        probs = probs.scatter_(1, state, torch.zeros_like(state, dtype=probs.dtype))

    dist = Categorical(probs=probs)

    action = dist.sample()

    return action.view((-1, 1))


def chain_dist_predict(model, state, k, trs=None):
    actions = list()
    for _ in range(k):
        action = direct_dist_predict(model, state, trs)
        actions.append(action)
        state = torch.cat([state[:, 1:], action], dim=1)
    prediction = torch.cat(actions, dim=1)
    return prediction


def prepare_true_matrix(tes, items_n, device):
    true_matrix = torch.zeros(
        size=(len(tes), items_n + 1), device=device
    )  # + padding index

    for i, te in enumerate(tes):
        true_matrix[i, te] = 1

    return true_matrix


def prepare_true_matrix(tes, rewards, items_n, device):
    true_matrix = torch.zeros(
        size=(len(tes), items_n + 1), device=device
    )  # + padding index

    for i, te in enumerate(tes):
        for j in range(len(te)):
            true_matrix[i, te[j]] = rewards[i][j]

    return true_matrix


def direct_predict_transformer(model, source, src_mask, padding_idx, trs=None):
    pad_mask = create_pad_mask(matrix=source, pad_token=padding_idx)
    output = model(src=source, src_mask=src_mask, src_key_padding_mask=pad_mask)

    output_last = output[:, :, -1]

    if trs:
        for i, tr in enumerate(trs):
            output_last[i, tr] = 0

    return output_last


def chain_predict_transformer(model, source, src_mask, padding_idx, k, trs=None):
    actions = list()
    for _ in range(k):
        output = direct_predict_transformer(
            model, source, src_mask, padding_idx, trs=trs
        )
        action = torch.argmax(output, dim=1, keepdim=True)
        actions.append(action)
        source = torch.cat([source[:, 1:], action], dim=1)
    prediction = torch.cat(actions, dim=1)
    return prediction
