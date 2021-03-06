import torch
import torch.nn.functional as F
import numpy as np
from models.transformer import create_pad_mask


def compute_td_loss(model, state, action, reward, next_state, done, gamma):
    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value).pow(2).mean()

    return loss


def compute_cql_loss(model, target_model, state, action, reward, next_state, done, gamma):
    with torch.no_grad():
        Q_targets_next = target_model(next_state).detach().max(1)[0]
        Q_targets = reward + gamma * Q_targets_next * (1 - done)
    Q_a_s = model(state)
    Q_expected = Q_a_s.gather(1, action.unsqueeze(1)).squeeze(1)

    bellmann_error = F.mse_loss(Q_expected, Q_targets)

    cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()

    q1_loss = cql1_loss + 0.5 * bellmann_error

    return q1_loss, cql1_loss, bellmann_error


def compute_td_loss_transformer_finetune(model, batch, gamma, src_mask, padding_idx):
    states, rewards, next_states, dones = batch

    pad_mask = create_pad_mask(matrix=states, pad_token=padding_idx)
    pad_mask_next = create_pad_mask(matrix=next_states, pad_token=padding_idx)

    q_values = model(states, src_mask=src_mask, src_key_padding_mask=pad_mask)
    next_q_values = model(
        next_states, src_mask=src_mask, src_key_padding_mask=pad_mask_next
    )
    q_value = q_values.gather(1, next_states.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    loss = (q_value - expected_q_value).pow(2).mean()

    return loss


def compute_td_loss_transformer(
    model, state, action, reward, next_state, done, gamma, src_mask, padding_idx
):
    pad_mask = create_pad_mask(matrix=state, pad_token=padding_idx)
    pad_mask_next = create_pad_mask(matrix=next_state, pad_token=padding_idx)

    output = model(state, src_mask=src_mask, src_key_padding_mask=pad_mask)
    q_values = output[:, :, -1]

    next_output = model(
        next_state, src_mask=src_mask, src_key_padding_mask=pad_mask_next
    )
    next_q_values = next_output[:, :, -1]

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value).pow(2).mean()

    return loss


def ddpg_loss(
    state,
    action,
    reward,
    next_state,
    done,
    value_net,
    policy_net,
    target_value_net,
    target_policy_net,
    value_criterion,
    gamma=0.99,
    min_value=-np.inf,
    max_value=np.inf,
):
    policy_output = policy_net(state)
    policy_loss = value_net(state, policy_output)
    policy_loss = -policy_loss.mean()

    reward = reward.unsqueeze(1)
    done = done.unsqueeze(1)

    target_policy_output = target_policy_net(next_state)
    target_value = target_value_net(next_state, target_policy_output.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    action_prob = torch.normal(
        0, 0.1, size=policy_output.size(), device=policy_output.device
    )
    for i, a in enumerate(action):
        action_prob[i, a] += 1
    action_prob = F.softmax(action_prob, dim=1)

    value = value_net(state, action_prob)
    value_loss = value_criterion(value, expected_value.detach())

    return policy_loss, value_loss


def compute_cql_loss_transformer(model, target_model, batch, gamma, src_mask, padding_idx):
    states, rewards, next_states, dones = batch

    pad_mask = create_pad_mask(matrix=states, pad_token=padding_idx)
    pad_mask_next = create_pad_mask(matrix=next_states, pad_token=padding_idx)

    with torch.no_grad():
        Q_targets_next = target_model(next_states, src_mask=src_mask, src_key_padding_mask=pad_mask_next).detach().max(1)[0]
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
    Q_a_s = model(states, src_mask=src_mask, src_key_padding_mask=pad_mask)
    Q_expected = Q_a_s.gather(1, next_states.unsqueeze(1)).squeeze(1)

    bellmann_error = F.mse_loss(Q_expected, Q_targets)

    cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()

    q1_loss = cql1_loss + 0.5 * bellmann_error

    return q1_loss, cql1_loss, bellmann_error


def compute_td_ce_loss_transformer(model, batch, gamma, src_mask, padding_idx, alpha):
    states, rewards, next_states, dones = batch

    pad_mask = create_pad_mask(matrix=states, pad_token=padding_idx)
    pad_mask_next = create_pad_mask(matrix=next_states, pad_token=padding_idx)

    q_values = model(states, src_mask=src_mask, src_key_padding_mask=pad_mask)
    next_q_values = model(
        next_states, src_mask=src_mask, src_key_padding_mask=pad_mask_next
    )
    q_value = q_values.gather(1, next_states.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    td_loss = (q_value - expected_q_value).pow(2).mean()

    ce = F.cross_entropy(q_values, next_states)

    loss = td_loss + alpha * ce

    return loss
