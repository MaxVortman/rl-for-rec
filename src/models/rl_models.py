import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        action_n: int,
        embedding_dim: int,
        seq_size: int,
        padding_idx: int = 0,
    ) -> None:
        super(DQN, self).__init__()

        self.action_embedding = nn.Embedding(
            action_n + 1, # + padding index 
            embedding_dim, 
            padding_idx=padding_idx
        )
        self.linears = nn.Sequential(
            nn.Linear(seq_size * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_n),
        )

    def forward(self, state):
        x = self.action_embedding(state)  # [B, S] -> [B, S, F]
        x = x.view(state.size()[0], -1)  # [B, S, F] -> [B, F]
        x = self.linears(x)
        return x
