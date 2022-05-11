import torch
import torch.nn.functional as F
import numpy as np


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
    policy_loss = value_net(state, policy_output)
    policy_loss = -policy_loss.mean()

    reward = reward.unsqueeze(1)
    done = done.unsqueeze(1)

    target_policy_output = target_policy_net(next_state)
    target_value = target_value_net(next_state, target_policy_output.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    action_prob = torch.normal(0, 0.1, size=policy_output.size(), device=policy_output.device)
    for i, a in enumerate(action):
        action_prob[i, a] += 1
    action_prob = F.softmax(action_prob, dim=1)

    value = value_net(state, action_prob)
    value_loss = value_criterion(value, expected_value.detach())

    return policy_loss, value_loss
