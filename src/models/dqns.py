import torch
import torch.nn as nn


class FixedFlatDQN(nn.Module):
    def __init__(
        self,
        action_n: int,
        seq_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        padding_idx: int = 0,
    ) -> None:
        super(FixedFlatDQN, self).__init__()

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

    def forward(self, state):
        x = self.action_embedding(state)  # [B, S] -> [B, S, F]
        x = x.view(state.size()[0], -1)  # [B, S, F] -> [B, S * F]
        x = self.linears(x)
        return x


class FixedAggsDQN(nn.Module):
    def __init__(
        self,
        action_n: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        padding_idx: int = 0,
    ) -> None:
        super(FixedAggsDQN, self).__init__()

        self.action_embedding = nn.Embedding(
            action_n + 1, embedding_dim, padding_idx=padding_idx  # + padding index
        )
        self.linears = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_n + 1),  # + padding index
        )
        self.padding_idx = padding_idx

    def forward(self, state):
        x = self.action_embedding(state)  # [B, S] -> [B, S, F]
        x_min, _ = torch.min(x, 1)  # [B, F]
        x_max, _ = torch.max(x, 1)  # [B, F]
        x_avg = torch.mean(x, 1)  # [B, F]
        x_last = x[:, -1]  # [B, F]
        x = torch.concat(
            [x_min, x_max, x_avg, x_last], dim=1
        )  # [B, S, F] -> [B, 4 * F]
        x = self.linears(x)
        return x
