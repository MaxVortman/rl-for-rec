import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        action_n: int,
        seq_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        padding_idx: int = 0,
    ) -> None:
        super(DQN, self).__init__()

        self.action_embedding = nn.Embedding(
            action_n + 1, embedding_dim, padding_idx=padding_idx  # + padding index
        )
        self.linears = nn.Sequential(
            nn.Linear(seq_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_n + 1),  # + padding index
        )
        self.padding_idx = padding_idx

    def forward(self, state):
        x = self.action_embedding(state)  # [B, S] -> [B, S, F]
        x = x.view(state.size()[0], -1)  # [B, S, F] -> [B, F]
        x = self.linears(x)
        x[:, self.padding_idx] = 0  # set padding index to 0
        return x
