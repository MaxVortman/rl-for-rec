import torch
import numpy as np
from .utils import soft_update


def compute_td_loss(model, state, action, reward, next_state, done, gamma):
    q_values = model(state)
    next_q_values = model(next_state)
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
    policy_action = torch.argmax(policy_output, keepdim=True, dim=1)
    policy_loss = value_net(state, policy_action)
    policy_loss = -policy_loss.mean()

    reward = reward.unsqueeze(1)
    done = done.unsqueeze(1)

    target_policy_output = target_policy_net(next_state)
    next_action = torch.argmax(target_policy_output, keepdim=True, dim=1)
    target_value = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)
    value = value_net(state, action.unsqueeze(1))
    value_loss = value_criterion(value, expected_value.detach())

    return policy_loss, value_loss
