def compute_td_loss(model, batch, gamma):
    state, action, reward, next_state, done = batch

    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value).pow(2).mean()

    return loss
